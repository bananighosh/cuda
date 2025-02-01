#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

void verify_result(float *a, float *b, float *c){
    float temp;
    float epsilon = 0.001;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            temp = 0;
            for(int k = 0; k < n; k++){
                temp += a[k * n + i] + b[j * n + k]; //column major order
            }
            assert(fabs(c[j * n + i] - temp) < epsilon); 
        }
    }
}

int main(){
    int n = 1 << 10;
    size_t bytes = n * n * sizeof(float);

    // declare the pointers to matrices on device to host
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // Allocate memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // initialize the vectors directly on GPU instead of CPU
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed
    curandSetPseudoGeneratorSeed(prng, (unsigned long long)clock());

    // Fill the matrix with random integers
    curandGenerateUniform(prng, d_a, n * n);
    curandGenerateUniform(prng, d_b, n * n)

    //cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    //scaling factor
    float alpha = 1.0f;
    float beta = 0.0f;

    // calculation: (alpha * a) * b + (beta * c)
    // ( m * n ) * (n * k) = (m * k)
    // cublasSgemm( handle, operation, operation, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, d_a, n, d_b, n, beta, d_c, n);

    //copy back all 3 matrices
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    verify_result(h_a, h_b, h_c, n);

    print("COMPLETED SUCCESSFULLY \n");

    return 0;
}