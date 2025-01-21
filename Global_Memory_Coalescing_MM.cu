#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N)) //needed for calculation of row and column index
#define M 256  // Number of rows in A and C
#define K 512  // Number of columns in A and rows in B
#define N 256  // Number of columns in B and C
#define BLOCK_SIZE 32

__global__ void GloMem_Coalesced_MM(float *A, float *B, float *C, int m, int k, int n, float alpha, float beta)
{
    const int row = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE); //map 1D thread index to a 2D coordinate space
    const int column = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

    if (row < m && column < n)
    {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i)
        {
            sum += A[row * k + i] * B[i * n + column];
        }
        //alpha used to scale current "sum", the product of matrix A & B, and beta used to scale previous "sum" (C = α*(AB)+β*C)
        C[row * n + column] = alpha * sum + beta * C[row * n + column]; 
    }
}

void gen_matrix(float *mat, int row, int column)
{
    for (int i = 0; i < row * column; i++)
    { 
        mat[i] = ((float)rand() / RAND_MAX); // Populate matrix with random values
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

    int size_A = M * K * sizeof(float); //calculate bytes needed for each matrix
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    cudaMalloc(&d_A, size_A);   //allocate space on GPU for each matrix
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    
    float *h_A = (float *)malloc(size_A); //allocate space on CPU for each matrix
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    gen_matrix(h_A, M, K); // Generate matrices on the host 
    gen_matrix(h_B, K, N);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE), 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        GloMem_Coalesced_MM<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N, alpha, beta);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        GloMem_Coalesced_MM<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N, alpha, beta);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Print results
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
