#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <iostream>

#define N 256  // Number of columns in B and C
#define TILE_SIZE 16 //smaller tile size speeds things up
#define BLOCK_SIZE 16

__global__ void tileMatMul(float *A, float *B, float *C, int n)
{
    int by = blockIdx.y; //initialize indexes as variables for simplicity
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    int row = by * blockDim.y + ty;
    int column = bx * blockDim.x + tx;

    __shared__ float shA[TILE_SIZE][TILE_SIZE]; //create shared memory for tile matrixes
    __shared__ float shB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    
    for(int tile_num = 0; tile_num < N/TILE_SIZE; tile_num++) //main algo
    {
        if((row < N) && ((tile_num * TILE_SIZE + tx) < N)) //if thread doesnt load a value in shared mem
        {
            //map global index to shared index for matrix A
            shA[ty][tx] = A[(row * N) + (tile_num * TILE_SIZE + tx)]; 
        }
        else
        {
            shA[ty][tx] = 0.0f;
        }


        if(((tile_num * TILE_SIZE + ty) < N) && (column < N))
        {
            //map global index to shared index for matrix B
            shB[ty][tx] = B[((tile_num * TILE_SIZE + ty) * N) + column]; 
        }
        else
        {
            shB[ty][tx] = 0.0f;
        }
        __syncthreads();

        for(int k = 0; k < TILE_SIZE; k++)
        {
            sum += shA[ty][k] * shB[k][tx];
        }
        __syncthreads();
    }  
    C[row * N + column] = sum;
    
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
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        tileMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        tileMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
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
