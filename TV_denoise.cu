/* Total Variation Denoising
 * using the over-relaxed Chambolle-Pock algorithm
 * an Cuda implementation
 * The function expects a bitmap image (*.ppm) as input.
 * Author: June Wu (qw262)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "noise.h"
#include "denoise.h"
extern "C" {
#include "ppma_io.h"
}


    int main(int argc, char *argv[])
{
    
    /****proximity, relaxation and regularizatio parameters ****/
    float lambda = 0.1, tau = 0.01, sigma = 1/tau/8, rho = 1.99;
    int Niter = 100;
    /**********************************************************/
    
    int error, xsize, ysize, rgb_max;
    int *r, *b, *g;

    double *noisyGray, *d_noisyGray;
    double *gray, *d_cost, cost = 0;
    
    /*********** denoising variables (are only on the device) **************/
    double *d_grad_row, *d_grad_col,*d_dual_col, *d_dual_row, *d_adj;
    double *d_diff, *d_temp_dual_row, *d_temp_dual_col, *d_temp_sol, *d_sol;


    if(argc != 2)
    {
        fprintf(stderr, "Usage: %s image.ppm \n", argv[0]);
        abort();
    }

    const char* filename = argv[1];

    // --------------------------------------------------------------------------
    // load image
    // --------------------------------------------------------------------------
    printf("Reading ``%s''\n", filename);
    ppma_read(filename, &xsize, &ysize, &rgb_max, &r, &g, &b);
    printf("Done reading ``%s'' of size %dx%d\n", filename, xsize, ysize);
    
    // --------------------------------------------------------------------------
    // allocate buffers on cpu and gpu
    // --------------------------------------------------------------------------
    gray = (double * ) calloc(xsize*ysize, sizeof(double));
    noisyGray = (double * ) calloc(xsize*ysize, sizeof(double));
    cudaMalloc((void ** ) &d_noisyGray, xsize*ysize*sizeof(double));
    cudaMalloc((void ** ) &d_cost, xsize*ysize*sizeof(double));
    
    cudaMalloc((void ** ) &d_sol, xsize*ysize*sizeof(double));
    cudaMalloc((void ** ) &d_temp_sol, xsize*ysize*sizeof(double));
    cudaMalloc((void ** ) &d_grad_row, xsize*ysize*sizeof(double));;
    cudaMalloc((void ** ) &d_grad_col, xsize*ysize*sizeof(double));
    cudaMalloc((void ** ) &d_dual_row, xsize*ysize*sizeof(double));
    cudaMalloc((void ** ) &d_dual_col, xsize*ysize*sizeof(double));
    cudaMalloc((void ** ) &d_diff, xsize*ysize*sizeof(double));
    cudaMalloc((void ** ) &d_temp_dual_row, xsize*ysize*sizeof(double));
    cudaMalloc((void ** ) &d_temp_dual_col, xsize*ysize*sizeof(double));
    cudaMalloc((void ** ) &d_adj, xsize*ysize*sizeof(double));

    // --------------------------------------------------------------------------
    // convert image to grayscale
    // --------------------------------------------------------------------------
    for(int n = 0; n < xsize*ysize; ++n){
        noisyGray[n] = (0.21f*r[n])/rgb_max + (0.72f*g[n])/rgb_max + (0.07f*b[n])/rgb_max;
    }
    
    // --------------------------------------------------------------------------
    // Copy grayscale image to device and add Gaussian white noise
    // --------------------------------------------------------------------------
    cudaMemcpy(d_noisyGray, noisyGray, xsize*ysize*sizeof(double), cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((int)ceil(xsize / 16.0), (int)ceil(ysize / 16.0));
    curandState* devStates;
    cudaMalloc (&devStates, xsize*ysize* sizeof(curandState));
    srand(time(0));
    int seed = rand();
    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(devStates,seed, xsize);
    noise<<<blocksPerGrid, threadsPerBlock>>>(d_noisyGray, xsize, ysize, d_cost, devStates);
    cudaMemcpy(noisyGray, d_noisyGray, xsize*ysize*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(noisyGray, d_temp_sol, xsize*ysize*sizeof(double), cudaMemcpyHostToDevice);
  
    // --------------------------------------------------------------------------
    // output the image that added Gaussian white noise
    // --------------------------------------------------------------------------
    printf("Writing noisy image\n");
    for(int n = 0; n < xsize*ysize; ++n){
        r[n] = g[n] = b[n] = (int)(noisyGray[n] * rgb_max);
    }
    error = ppma_write("noisy_image.ppm", xsize, ysize, r, g, b);
    if(error) { fprintf(stderr, "error writing image"); abort(); }

    // --------------------------------------------------------------------------
    // Denoise using total variation
    // --------------------------------------------------------------------------
    /* Compute gradient row-wise and column-wise */
    gradient<<<blocksPerGrid, threadsPerBlock>>>(d_noisyGray, d_grad_row, d_grad_col, xsize, ysize);

    /* Initialize the dual solution */
    conj<<<blocksPerGrid, threadsPerBlock>>>(d_temp_dual_row, d_temp_dual_col, d_grad_row, d_grad_col, xsize, ysize, lambda);
    
    /*Compute total cost */
    double *sum, *d_sum;
    int blockSize = (int)ceil(xsize*ysize / 256.0);
    sum = (double *) calloc(blockSize, sizeof(double));
    cudaMalloc((void ** ) &d_sum, blockSize*sizeof(double));
    sum_reduction<<<blockSize, 256>>>(d_cost, d_sum, xsize*ysize);
    cudaMemcpy(sum, d_sum, blockSize*sizeof(double), cudaMemcpyDeviceToHost);
    int count;
    for (count = 0; count < blockSize; count++){
        cost = cost + sum[count];
    }
    cost = cost / 2;
    
    printf("The total cost is %lf \n", cost);
    
    for(count = 0; count < Niter; count++){
        adjoint<<<blocksPerGrid, threadsPerBlock>>>(d_adj, d_temp_dual_row, d_temp_dual_col, xsize, ysize);
        
        /* compute current solution */
        prox_tau_f<<<blocksPerGrid, threadsPerBlock>>>(d_adj, d_sol, d_temp_sol, d_noisyGray, xsize, ysize, tau);
        
        /* compute current dual solution */
        diff<<<blocksPerGrid, threadsPerBlock>>>(d_sol, d_temp_sol, d_diff, xsize, ysize);
        gradient<<<blocksPerGrid, threadsPerBlock>>>(d_diff, d_grad_row, d_grad_col, xsize, ysize);
        ascent<<<blocksPerGrid, threadsPerBlock>>>(d_temp_dual_row, d_temp_dual_col, d_grad_row, d_grad_col, xsize, ysize, sigma);
        conj<<<blocksPerGrid, threadsPerBlock>>>(d_dual_row, d_dual_col, d_grad_row, d_grad_col, xsize, ysize, lambda);
        update_sol<<<blocksPerGrid, threadsPerBlock>>>(d_temp_sol, d_sol, xsize, ysize, rho);
        upadate_dual<<<blocksPerGrid, threadsPerBlock>>>(d_temp_dual_row, d_temp_dual_col, d_dual_row, d_dual_col, xsize, ysize, rho);
        
    }
    
    cudaMemcpy(gray, d_sol, xsize*ysize*sizeof(double), cudaMemcpyDeviceToHost);
    // --------------------------------------------------------------------------
    // output denoised image
    // --------------------------------------------------------------------------
    printf("Writing denoised image\n");
    for(int n = 0; n < xsize*ysize; ++n){
        r[n] = g[n] = b[n] = (int)(gray[n] * rgb_max);
    }
    error = ppma_write("output.ppm", xsize, ysize, r, g, b);
    if(error) { fprintf(stderr, "error writing image"); abort(); }
    
    // --------------------------------------------------------------------------
    // clean up
    // --------------------------------------------------------------------------
    cudaFree(d_noisyGray);
    cudaFree(d_cost);
    cudaFree(d_dual_col);
    cudaFree(d_dual_row);
    cudaFree(d_temp_dual_col);
    cudaFree(d_temp_dual_row);
    cudaFree(d_grad_col);
    cudaFree(d_grad_row);
    cudaFree(d_diff);
    cudaFree(d_sum);
    cudaFree(d_temp_sol);
    cudaFree(d_sol);
    free(noisyGray);
    free(r);
    free(b);
    free(g);
    free(gray);
    free(sum);
    
    return 0;

}


