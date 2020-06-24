#include <iostream>
#include <chrono> 

using namespace std::chrono; 
using namespace std;

#define n (1 << 10)

__global__ void matrix_multiplication_kernel(int *d_a, int *d_b, int *d_c){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= n || j >= n) return;

    d_c[i*n + j] = 0;
    for(int k=0; k<n; k++){
        d_c[i*n+j] += d_a[i*n+k] * d_b[k*n+j];
    }
}

int main(){

    size_t bytes = n*n*sizeof(int);

    int *h_a, *h_b, *h_c;
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    for(int i = 0; i < n*n; i++){
        h_a[i] = i;
        h_b[i] = i;
    }

    // for(int i = 0; i < n; i++){
    //     for(int j=0; j<n; j++){
    //         cout << h_a[i*n + j] << "\t";
    //     }
    //     cout << endl;
    // }

    // cout << "*" << endl;

    // for(int i = 0; i < n; i++){
    //     for(int j=0; j<n; j++){
    //         cout << h_b[i*n + j] << "\t";
    //     }
    //     cout << endl;
    // }

    cout << "cpu: " << endl;
    auto start = high_resolution_clock::now();     
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            h_c[i*n + j] = 0;
            for(int k=0; k<n; k++){
                h_c[i*n+j] += h_a[i*n+k] * h_b[k*n+j];
            }
        }
    }
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "cpu time: " << duration.count() << endl; 
    
    // for(int i = 0; i < n; i++){
    //     for(int j=0; j<n; j++){
    //         cout << h_c[i*n + j] << "\t";
    //     }
    //     cout << endl;
    // }

    cout << "gpu: " << endl;
    start = high_resolution_clock::now();
    int *h_d = (int*)malloc(bytes);

    int *d_a, *d_b, *d_d;
    cudaMalloc((void**) &d_a, bytes);
    cudaMalloc((void**) &d_b, bytes);
    cudaMalloc((void**) &d_d, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 32;
    int GRID_SIZE = (n/BLOCK_SIZE) + 1;
    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    matrix_multiplication_kernel<<<grid,block>>>(d_a,d_b,d_d);

    cudaMemcpy(h_d, d_d, bytes, cudaMemcpyDeviceToHost);
    
    // for(int i = 0; i < n; i++){
    //     for(int j=0; j<n; j++){
    //         cout << h_d[i*n + j] << "\t";
    //     }
    //     cout << endl;
    // }

    stop = high_resolution_clock::now(); 
    duration = duration_cast<microseconds>(stop - start);
    cout << "gpu time: " << duration.count() << endl; 

    bool error_occurred = false;
    for(int i = 0; i < n; i++){
        for(int j=0; j<n; j++){
            if(h_d[i*n + j] - h_c[i*n + j] != 0){
                cout << "Some error occurred" <<endl;
                error_occurred = true;
            }
        }
    }
    if(error_occurred == false) cout << "No error" <<endl;
}