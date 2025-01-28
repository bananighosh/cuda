#include <studio.h>
#include <math.h>
#include <assert.h>

// GPU CODE
// CUDA kernal for vector addition
__global__ void vectorAdd(int* a, int* b, int* c, int n){
    //Calculate the global threadid
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Vector boundary guard
    if (tid < n){
        c[tid] = a[tid] + b[tid];
    }
}

// Initialize vector of size  n in between 0-99
void matrix_init(int* a, int n){
    for(int i = 0; i < n; i++){
        a[i] = rand() % 100
    }
}

// main
int main(){
    // vector size 2 ^ 16
    int n = 1 << 16;

    // host vector pointers
    int *h_a, *h_b, *h_c;

    // device vector pointers
    int *d_a, *d_b, *d_c;

    // Allocation size for all vectors
    size_t bytes = sizeof(int) * n;

    // Allocate host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialize vectors a and b with random values between 0 and 99
    matrix_init(h_a, n);
    matrix_init(h_b, n);

    // Copy data n
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Threadblock size
    int NUM_THREADS = 256;

    // Grid size
    int NUM_BLOCKS = (int)ceil(n / NUM_THREADS);

    // Launch Kernel on default stream w/o shared memory(shmem)
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);

    // Copy sum vector from device to host
    cudaMemcpy(h_c, d_c, bytes, cudeMemcpyDeviceToHost);

    // Check result for errors
    error_check(h_a, h_b, _h_c, n);

    printf("COMPLETED SUCCESSFULLY \n");

    return 0;
}