
// --------------------------------------------------------------------------
// Cuda random number generator
// --------------------------------------------------------------------------
__device__ float generate(curandState* globalState, int ind)
{
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

__global__ void setup_kernel (curandState * state, unsigned long seed, int xsize)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int id = i*xsize + j;
    curand_init ( seed, id, 0, &state[id] );
}

// --------------------------------------------------------------------------
// Kernel that adds Gaussian white noise
// --------------------------------------------------------------------------
__global__ void noise(double *d_noisyGray, int xsize, int ysize, double *d_cost, curandState* globalState){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    double noise;
     if((i < ysize) && (j < xsize)){
        noise = d_noisyGray[i * xsize + j] + generate(globalState, i*xsize + j)* 0.3;
        d_cost[i*xsize + j] = noise * noise;
        d_noisyGray[i*xsize + j] = (float) noise;
     }
    
}
