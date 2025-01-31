#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Initialize a vector
void vector_init(float* a, int n){
    for(int i = 0; i < n; i++){
        a[i] = (float)(rand() % 100)
    }
}

// verify the result
void verify_result(float* a, float* b, float* c, float factor, int n){
    for(i = 0; i < n; i++){
        assert(c[i] == factor * a[i] + b[i])
    }
}

int main(){
    int n = 1 << 2;
     size_t bytes = n * sizeof(float);

     // Dec;are vector pointers
     float* h_a, h_b, h_c;
     float* d_a, d_b;

     // Allocate memory
     h_a = (float)malloc(bytes);
     h_b = (float)malloc(bytes);
     h_c = (float)malloc(bytes);
     cudaMalloc(&d_a, bytes);
     cudaMalloc(&d_b, bytes);

     // Initialize vectors
     vector_init(h_a, n);
     vector_init(h_b, n);

     // Create and initialize the cublas handle ( or a new context)
     cublasHandle_t handle;
     cublasCreate_v2(handle);

     // Copy the vectors over to device
     cublasSetVector(n, sizeof(float), h_a, 1, d_a, 1);
     cublasSetVector(n, sizeof(float), h_b, 1, d_b, 1);

     //Launch a simple saxpy kernel ( single precision a * x + y)
     const float scale = 2.0f;
     cublasSaxpy(handle, n, sizeof(float), d_a, 1, d_b, 1); // 1 represents our vectors doesnot have any empty spaces to skip

     // copy result to host
     cublasSetVector(n, sizeof(float), d_b, 1, h_c, 1 );

     // verify 
     verify_result(h_a, h_b, h_c, scale, n);

     // clean up the cublas handle
     cublasDestroy(handle)

     // free up memory
     cudaFree(d_a);
     cudaFree(d_b);
     free(h_a);
     free(h_b);

     return 0;
}