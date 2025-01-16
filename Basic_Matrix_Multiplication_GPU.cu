#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256  // Number of rows in A and C
#define K 512  // Number of columns in A and rows in B
#define N 256  // Number of columns in B and C
#define BLOCK_SIZE 32

__global__ void matrixMultiplication_GPU(float *A, float *B, float *C, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < m && column < n)
    {
        float sum = 0.0f;
        for(int l = 0; l < k; l++)
        {
            sum += A[row * k + l] * B[l * n + column];
        }
        C[row * n + column] = sum;
        //printf("C[%d][%d] = %f\n", row, column, sum);

    }  
}

void gen_matrix(float *mat, int row, int column)
{
    for(int i = 0; i < row * column; i++)
    { 
        mat[i] = ((float)rand() / RAND_MAX); // Populate matrix with values from 1 to 10
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
    
    int size_A = M * K * sizeof(float); //calculate bytes needed for each matrix
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);
    
    cudaMalloc(&d_A, size_A);   //allocate space for each matrix
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Generate matrices on the host (in this case d_A and d_B are device pointers, but gen_matrix is not designed for device memory)
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    
    gen_matrix(h_A, M, K); 
    gen_matrix(h_B, K, N);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        matrixMultiplication_GPU<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matrixMultiplication_GPU<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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
