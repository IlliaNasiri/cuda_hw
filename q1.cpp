#include "utils.h"

__global__ void matmul_thread_per_element(
    const float* A, // dimensions: [M, K]
    const float* B, // dimensions: [K, N]
    float* C, // dimensions: [M, N]
    const unsigned int M,
    const unsigned int K,
    const unsigned int N
    )
{
    unsigned int row = threadIdx.x + (blockIdx.x * blockDim.x);
    unsigned int col = threadIdx.y + (blockIdx.y * blockDim.y);
    if (row < M && col < N) {
        float sum = 0.0f;
        for (unsigned int i = 0; i < K; i++)
            // C[row][col] = sum_{i=0}^{i=K} ( A_{row, i} * B_{i, col}
                sum += A[row * K + i] * B[i * N + col];

        C[row * N + col] = sum;
    }

}


float block_size_experiments(
    unsigned int M,
    unsigned int K,
    unsigned int N,
    dim3 grid_size,
    dim3 block_size
)
{
    // Memory is allocated and freed on both RAM and GPU automatically
    // when creating HeterogenousMatrix class.
    HeterogeneousMatrix A(M, K);
    HeterogeneousMatrix B(K, N);
    HeterogeneousMatrix C(M, N);

    // Generate input matrices:
    initialize_matrix_hw2(A);
    initialize_matrix_hw2(B);

    A.to_device();
    B.to_device();

    // Timing done based on the following tutorial https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_thread_per_element<<<grid_size, block_size>>>(A.d_data, B.d_data, C.d_data, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    C.to_host();

    std::cout << "C(0,0) = " << C.h_data[0] << '\n'
              << "C(0,1) = " << C.h_data[1] << '\n'
              << "C(1,0) = " << C.h_data[C.n_cols] << '\n'
              << "C(1,1) = " << C.h_data[C.n_cols + 1] << '\n';


    return milliseconds;
}


int main() {
    unsigned int M = 20000;
    unsigned int K = 20000;
    unsigned int N = 20000;

    dim3 block_sizes[] = {
        dim3(32, 32, 1),
        dim3(16, 16, 1),
        dim3(32, 16, 1),
        dim3(8, 8, 1)
    };

    for (auto block_size : block_sizes) {
        //
        float block_x = block_size.x, block_y = block_size.y;
        dim3 grid_size( ceil(M / block_x), ceil(N / block_y) );

        std::cout << "block size: " << dim3_to_string(block_size) << " , grid_size" << dim3_to_string(grid_size) << std::endl;

        float elapsed_time = block_size_experiments(M, K, N, grid_size, block_size);
        std::cout << "kernel elapsed time: " << elapsed_time << "ms" << std::endl;

        std::cout << std::endl << "--------" << std::endl;

    }

}
