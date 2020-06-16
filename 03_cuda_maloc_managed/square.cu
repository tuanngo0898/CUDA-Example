#include <iostream>
using namespace std;

__global__ void square(int *d_out, int *d_in){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int i = d_in[idx];
    d_out[idx] = i*i;
}

int main(){
    const int ARRAY_SIZE = 1000;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
    int id = cudaGetDevice(&id);

    int *in, *out;

    cudaMallocManaged((void**)&in, ARRAY_BYTES);
    cudaMallocManaged((void**)&out, ARRAY_BYTES);

    for(int i=0; i < ARRAY_SIZE; i++){
        in[i] = i;
    }

    int NUM_THREADS = 512;
    int NUM_BLOCKS  = ARRAY_SIZE / NUM_THREADS + 1;
    cudaMemPrefetchAsync(in, ARRAY_BYTES, id);
    square<<<NUM_BLOCKS, NUM_THREADS>>>(out, in);

    cudaDeviceSynchronize();
    cudaMemPrefetchAsync(out, ARRAY_BYTES, cudaCpuDeviceId);

    for(int i=0; i< ARRAY_SIZE; i++){
        cout << out[i];
        if(i%10!=9) cout << "\t";
        else cout << endl;
    }
}