#include <iostream>
using namespace std;

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__global__ void square(float *d_out, float *d_in){
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f*f;
}

int main(){
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    float h_in[ARRAY_SIZE];
    for(int i=0; i < ARRAY_SIZE; i++){
        h_in[i] = float(i);
    }
    float h_out[ARRAY_SIZE];

    float *d_in;
    float *d_out;

    CUDA_CALL(cudaMalloc((void**) &d_in, ARRAY_BYTES));
    CUDA_CALL(cudaMalloc((void**) &d_out, ARRAY_BYTES));

    CUDA_CALL(cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice));
    square<<<1, ARRAY_SIZE>>>(d_out, d_in);

    CUDA_CALL(cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));

    for(int i=0; i< ARRAY_SIZE; i++){
        cout << h_out[i];
        if(i%4!=3) cout << "\t";
        else cout << endl;
    }
}