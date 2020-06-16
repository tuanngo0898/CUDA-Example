#include <iostream>
using namespace std;

__global__ void square(float *d_out, float *d_in){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f*f;
}

int main(){
    const int ARRAY_SIZE = 10000;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    float h_in[ARRAY_SIZE];
    for(int i=0; i < ARRAY_SIZE; i++){
        h_in[i] = float(i);
    }
    float h_out[ARRAY_SIZE];

    float *d_in;
    float *d_out;

    cudaMalloc((void**) &d_in, ARRAY_BYTES);
    cudaMalloc((void**) &d_out, ARRAY_BYTES);

    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
    
    int NUM_THREADS = 512;
    int NUM_BLOCKS  = ARRAY_SIZE / NUM_THREADS + 1;
    square<<<NUM_BLOCKS, NUM_THREADS>>>(d_out, d_in);

    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    for(int i=0; i< ARRAY_SIZE; i++){
        cout << h_out[i];
        if(i%10!=9) cout << "\t";
        else cout << endl;
    }
}