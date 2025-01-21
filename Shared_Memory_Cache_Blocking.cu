#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define M 256  // Number of rows in A and C
#define K 512  // Number of columns in A and rows in B
#define N 256  // Number of columns in B and C
#define BLOCK_SIZE 32

__global__ void ShMem_Cache_Block_MM(float *A, float *B, float *C, int m, int k, int n, float alpha, float beta)
{
    int by = blockIdx.y; //initialize indexes as variables for simplicity
    int bx = blockIdx.x;

    int tRow = threadIdx.x / BLOCK_SIZE;
    int tCol = threadIdx.x % BLOCK_SIZE;
    
    __shared__ float shA[BLOCK_SIZE * BLOCK_SIZE]; //create shared memory for tile matrixes
    __shared__ float shB[BLOCK_SIZE * BLOCK_SIZE];

    A += by * BLOCK_SIZE * K;                    //dividing up each matrix into blocks
    B += bx * BLOCK_SIZE;                        
    C += by * BLOCK_SIZE * N + bx * BLOCK_SIZE; 
    
    float sum = 0.0f;
    
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE) 
    {
        shA[tRow * BLOCK_SIZE + tCol] = A[tRow * K + tCol];
        shB[tRow * BLOCK_SIZE + tCol] = B[tRow * N + tCol];

        // block threads in this block until cache is fully populated
        __syncthreads();
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) // peeform dotproduct on the cached block
        {
            sum += shA[tRow * BLOCK_SIZE + dotIdx] * shB[dotIdx * BLOCK_SIZE + tCol]; 
        }
    __syncthreads();
  }
 
  C[tRow * N + tCol] = alpha * sum + beta * C[tRow * N + tCol];
}

void gen_matrix(float *mat, int row, int column) //populates matrix with values
{
    for(int i = 0; i < row * column; i++)
    { 
        mat[i] = ((float)rand() / RAND_MAX);
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main()
{
    float *d_A, *d_B, *d_C; //initialize device matrices
    float alpha = 1.0f, beta = 0.0f;
    int size_matrix = N * N * sizeof(float); //calculate bytes needed for each matrix
  
    cudaMalloc(&d_A, size_matrix);   //allocate space on device (GPU)
    cudaMalloc(&d_B, size_matrix);
    cudaMalloc(&d_C, size_matrix);

    float *h_A = (float *)malloc(size_matrix); //allocate space on host (CPU)
    float *h_B = (float *)malloc(size_matrix);
    
    gen_matrix(h_A, N, N); //populates matrix A and B with random values
    gen_matrix(h_B, N, N);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size_matrix, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_matrix, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE), 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        ShMem_Cache_Block_MM<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N, alpha, beta);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        ShMem_Cache_Block_MM<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N, alpha, beta);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Print results
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));

    // Free host memory
    free(h_A);
    free(h_B);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
