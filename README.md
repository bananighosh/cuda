# cuda
CUDA 100 days of Programming

mentor: https://github.com/hkproj/
challenge:  https://github.com/bananighosh/100-days-of-gpu 

## Day 01:
- Coded the 1st gpu kernel with CUDA
- Learnt about SIMT and SIMD
- How GPUs work
- How data is shared between cpu and gpu
- Memory allocation and management using cuda library functions: cudaMalloc, cudaMemcpy 

## Day 02:
- What is Unified Memory
- How it helps in performance optimization
- Coded Vector addition with unified memory

## Day 03:
- Coded Kernal for Basic matrix multiplication
- Learnt how blocks , grids and threads are handled in a 2D vector
- Memory allocation and management using cuda library functions: cudaFree

## Day 04:
- What is CuBlas? Why to use it? 
    - runtime library
    - Blas: Basic 
-  vector add using CuBlas

## Day 05:
- matrix multiplication using `cublasSgemm` library
- row-major ordering vs column-major ordering
    - implemented column major ordering
    - compared the performance with row-major ordering
- Performance optimization by  initializing the  matrix directly on GPU avoiding data transfer from cpu
    - cudaRand libraries learnt: curandCreateGenerator, cudarandSetseudoGeneratorSeed, cudaRandGenerateUniform
