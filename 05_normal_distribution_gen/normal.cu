#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <math.h>

#define SHIFT       50
#define SCALE       5

#define ARRAY_SIZE  1000
#define BLOCK_SIZE  512

#define CUDA_CALL(ans) { GpuAssert((ans), __FILE__, __LINE__); }
inline void GpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      printf("GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void normal_init_kernel(curandState *state, float seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets different seed, a different sequence number, no offset */
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void normal_generate_kernel(curandState *state, float *result)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    curandState localState = state[idx];                            /* Copy state to local memory for efficiency */
    result[idx] = (curand_normal(&localState) * SCALE)+SHIFT;       /* Generate pseudo-random uniforms */
    state[idx] = localState;                                        /* Copy state back to global memory */
}

void normal_generator(float seed){
    curandState *dev_states;
    float *dev_array, *hst_array;

    hst_array = (float *)malloc(ARRAY_SIZE * sizeof(float));
    CUDA_CALL(cudaMalloc((void **)&dev_array, ARRAY_SIZE *sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&dev_states, ARRAY_SIZE * sizeof(curandState)));

    int grid_size = ARRAY_SIZE / BLOCK_SIZE + 1;
    normal_init_kernel<<<grid_size, BLOCK_SIZE>>>(dev_states, seed);
    normal_generate_kernel<<<grid_size, BLOCK_SIZE>>>(dev_states, dev_array);
    
    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hst_array, dev_array, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    /* Show result */
    int p[6*SCALE]={};
    for (int i=0; i<ARRAY_SIZE; i++) {
        if(hst_array[i] >= SHIFT-3*SCALE && SHIFT-SCALE < SHIFT+3*SCALE){
            p[(int)hst_array[i] - (SHIFT-3*SCALE)] ++;
        }
    }
    for (int i=SHIFT-3*SCALE; i<SHIFT+3*SCALE; ++i) {
        std::cout << i << "-" << (i+1) << ":";
        std::cout << "  " << std::string(p[i - (SHIFT-3*SCALE)],'*') << std::endl;
    }
    
    float total = 0;
    for(int i = 0; i < ARRAY_SIZE; i++) {
        total += hst_array[i];
    }
    printf("seed: %f, Results mean = %f\n", seed,(total/(1.0*ARRAY_SIZE)));

    /* Cleanup */
    CUDA_CALL(cudaFree(dev_array));
    CUDA_CALL(cudaFree(dev_states));
    free(hst_array);

    cudaDeviceSynchronize();
}


int main(int argc, char *argv[])
{
    int device;
    struct cudaDeviceProp properties;

    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaGetDeviceProperties(&properties,device));

    for(int i=0; i< 10; i++){
        normal_generator(i);
    }

    return 0;
}