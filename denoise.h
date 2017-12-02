// --------------------------------------------------------------------------
// Kernel that performs sum reduction
// --------------------------------------------------------------------------
__global__ void sum_reduction(double *d_vec, double *d_sum, int N){
    __shared__ double partial_sum[256];
    /* colaboratively read the value into partial_sum */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;
    partial_sum[t] = d_vec[i];
    __syncthreads();
    /* perform reduction within a block*/
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(t < stride && i < N){
            partial_sum[t] += partial_sum[t+stride];
        }
        __syncthreads();
    }
    if (t == 0){
        d_sum[blockIdx.x] = partial_sum[0];
    }
}

// --------------------------------------------------------------------------
// Compute gradient row-wise and column-wise
// --------------------------------------------------------------------------

__global__ void gradient(double *d_noisyGray, double *d_grad_row, double *d_grad_col, int xsize, int ysize){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    /*compute gradient by row */
    if ((i < ysize - 1) && (j < xsize))
        d_grad_row[i*xsize + j] = d_noisyGray[i*xsize+j] - d_noisyGray[(i+1)*xsize + j];
    if ((i == ysize - 1) && (j < xsize))
        d_grad_row[i*xsize + j] = 0;
    
    /*compute gradient by column */
    if ((i < ysize) && (j < xsize - 1))
        d_grad_col[i*xsize + j] = d_noisyGray[i*xsize+j] - d_noisyGray[i*xsize + j + 1];
    if ((i < ysize) && (j == xsize - 1))
        d_grad_col[i*xsize + j] = 0;
    
}

// --------------------------------------------------------------------------
// Compute conj row-wise and column-wise
// --------------------------------------------------------------------------

__global__ void conj(double *d_temp_dual_row, double *d_temp_dual_col, double *d_grad_row, double *d_grad_col, int xsize, int ysize, float lambda){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    double sum;
    if ((i < ysize) && (j < xsize)){
        sum = (d_grad_col[i*xsize + j])*(d_grad_col[i*xsize + j]) + (d_grad_row[i*xsize + j]) * (d_grad_row[i*xsize + j]);
        if ((sqrt(sum) / lambda - 1) > 0){
            d_temp_dual_col[i*xsize + j] = d_grad_col[i*xsize + j] / (sqrt(sum) / lambda);
            d_temp_dual_row[i*xsize + j] = d_grad_row[i*xsize + j] / (sqrt(sum) / lambda);
        }
        else{
            d_temp_dual_col[i*xsize + j] = d_grad_col[i*xsize + j]; 
            d_temp_dual_row[i*xsize + j] = d_grad_row[i*xsize + j];
        }
    }  
    
}
    
// --------------------------------------------------------------------------
// Compute adj 
// --------------------------------------------------------------------------
  
__global__ void  adjoint(double *d_adj, double *d_dual_row, double *d_dual_col, int xsize, int ysize){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < ysize - 1) && (j < xsize))
        d_adj[(i+1)*xsize + j] = -(d_dual_row[i*xsize+j] - d_dual_row[(i+1)*xsize + j]);
    if ((i == 0) && (j < xsize))
        d_adj[i*xsize + j] = -d_dual_row[i*xsize + j];
    if ((i < ysize) && (j < xsize - 1))
        d_adj[i*xsize + j + 1] = d_adj[i*xsize + j + 1] - (d_dual_col[i*xsize+j] - d_dual_col[i*xsize + j + 1]);
    if ((i < ysize) && (j == 0))
        d_adj[i*xsize + j] = d_adj[i*xsize + j] - d_dual_col[i*xsize + j];
}   
    
__global__ void prox_tau_f(double *d_adj, double *d_sol, double *d_temp_sol, double *d_noisyGray, int xsize, int ysize, float tau){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < ysize) && (j < xsize)){
        d_sol[i*xsize + j]  = ((d_temp_sol[i*xsize + j] - tau*d_adj[i*xsize+j]) + tau*d_noisyGray[i*xsize+j] ) / (1 + tau);
    }
}

__global__ void diff(double * d_sol, double * d_temp_sol, double *d_diff, int xsize, int ysize){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < ysize) && (j < xsize)){
        d_diff[i*xsize + j]  = 2 * d_sol[i*xsize + j] - d_temp_sol[i*xsize + j]; 
    }
}
    
    
__global__ void ascent(double *d_dual_row, double *d_dual_col, double *d_grad_row, double * d_grad_col, int xsize, int ysize, float sigma){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < ysize) && (j < xsize)){
        d_grad_row[i*xsize + j]  = d_dual_row[i*xsize + j] + sigma * d_grad_row[i*xsize + j]; 
        d_grad_col[i*xsize + j]  = d_dual_col[i*xsize + j] + sigma * d_grad_col[i*xsize + j]; 
    }
}

__global__ void update_sol(double * d_temp_sol, double * d_sol, int xsize, int ysize, float rho){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < ysize) && (j < xsize)){
    d_temp_sol[i*xsize + j]  = d_temp_sol[i*xsize + j] + rho * (d_sol[i*xsize + j] - d_temp_sol[i*xsize + j]);  
    }
}

__global__ void upadate_dual(double *d_temp_dual_row, double *d_temp_dual_col, double *d_dual_row, double *d_dual_col,int xsize, int ysize, float rho){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < ysize) && (j < xsize)){
    d_temp_dual_row[i*xsize + j]  = d_temp_dual_row[i*xsize + j] + rho * (d_dual_row[i*xsize + j] - d_temp_dual_row[i*xsize + j]);  
    d_temp_dual_col[i*xsize + j]  = d_temp_dual_col[i*xsize + j] + rho * (d_dual_col[i*xsize + j] - d_temp_dual_col[i*xsize + j]);  
    }
}

