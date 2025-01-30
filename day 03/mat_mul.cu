#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

// CUDA Kernel for matrix multiplication
__global__ void matrixMul(const int* a, const int* b, int* c, int N) {
    // Computing each thread's global row and column
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int col = (blockIdx.x * blockDim.y) + threadIdx.x;

    // Iterate over row and then down col
    c[row * N + col] = 0;
    for(int k = 0; k < N; k++){
        // accumulating results for a single element
        c[row * N + col] = a[row * N + k] * b[k * N + col] 
    }
}

// To check the result on CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N){
    // for every row
    for(int i = 0; i < N; i++){
        // for every col
        for(int j = 0; j < N; j++){
            int tmp = 0;
            // for evert ele in the row-col pair
            for(int k = 0; k < N; k++){
                    // row * N + k, k * N + col
                tmp += a[i * N + k] * b[k * N + j]
            }
        }
        // curr_ele: [row * N + col]
        assert(tmp == c[i * N + j])
    }
}

int main(){
    // matrix size 1024 * 1024 (2 ^ 10 * 2 ^ 10)
    int N = 1 << 10;

    size_t bytes = N * N * sizeof(int);

    // Host vectors
    vector<int> h_a(N * N);
    vector<int> h_b(N * N);
    vector<int> h_c(N * N);

    // Initialize matrices
    generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });
    
    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_a.data(), bytes, cudaMemcpyHostToDevice);

    int THREADS = 32;
    int BLOCKS = N / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // Launch Kernel
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);

    // Copy back to the host
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // check the result
    verify_result(h_a, h_b, h_c, N);

    cout << "COMPLETED SUCCESFULLY\n";

    // Free memory on device
    cudeFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}