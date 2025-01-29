// Vector Addition Unified Memory
#include <studio.h>
#include <math.h>
#include <assert.h>

__global__ vector_add_um(int* a, int* b, int* c, int n){

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < n){
        c[tid] = a[tid] + b[tid];
    }
}

void init_vector(int* a, int* b, int n){
    for(int i = 0; i < n; i++){
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }
}

void check_answer(int* a, int* b, int* c, int n){
    for(int i = 0; i < n; i++){
        assert(c[i] == a[i] + b[i]);
    }
}

int main(){
    // Get device ID for other CUDA calls
    int id = cudaGetDevice(&id);

    int n = 1 << 16;

    int *a, *b, *c;

    size_t bytes = n * sizeof(int);

    // automatically manages the data to be on CPU and GPU as and when necessary during execution
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    init_vector(a, b, n);

    // 256 threads per block
    int BLOCK_SIZE = 256;

    int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);

    // for prefetching 'a' and 'b' vectors to device
    // cudaMemPrefetchAsync(a, bytes, id);
    // cudaMemPrefetchAsync(b, bytes, id);

    // Launch kernel
    vector_add_um<<GRID_SIZE, BLOCK_SIZE>>(a, b, c, n);

    // wait for all prev operations are completed after this point
    cudaDeviceSynchronize();

    check_answer(a, b, c, n);

    print("COMPLETED SUCCESSFULLY\n");

    return 0;
}



}