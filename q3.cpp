#include "utils.h"


__global__ void matmul_thread_per_column(
    const float* A, // dimensions: [M, K]
    const float* B, // dimensions: [K, N]
    float* C, // dimensions: [M, N]
    const unsigned int M,
    const unsigned int K,
    const unsigned int N
)
{
    unsigned int thread_column = threadIdx.y + (blockDim.y * blockIdx.y);
    if (thread_column < N) {

        for (unsigned int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (unsigned int j = 0; j < K; j++)
                // A[i][j] * B[j][th_col]
                sum += A[ i * K + j ] * B[j * N + thread_column];

            C[i * N + thread_column] = sum;
        }

    }

}

float block_size_experiments(
    unsigned int M,
    unsigned int K,
    unsigned int N,
    dim3 grid_size,
    dim3 block_size
) {
    // Memory is allocated and freed on both RAM and GPU automatically
    // when creating HeterogenousMatrix class.
    HeterogeneousMatrix A(M, K);
    HeterogeneousMatrix B(K, N);
    HeterogeneousMatrix C(M, N);

    initialize_matrix_hw2(A);
    initialize_matrix_hw2(B);

    A.to_device();
    B.to_device();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_thread_per_column<<< grid_size, block_size >>>(A.d_data, B.d_data, C.d_data, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    C.to_host();

    return milliseconds;

}

int main() {
    unsigned int M = 20000;
    unsigned int K = 20000;
    unsigned int N = 20000;

    dim3 block_sizes[] = {
        dim3(1, 32, 1),
        dim3(1, 16, 1),
        dim3(1, 8, 1)
    };

    for (auto block_size : block_sizes) {
        //
        float block_x = block_size.x, block_y = block_size.y;
        dim3 grid_size( 1, ceil(M / block_x), 1 );

        std::cout << "block size: " << dim3_to_string(block_size) << " , grid_size" << dim3_to_string(grid_size) << std::endl;

        float elapsed_time = block_size_experiments(M, K, N, grid_size, block_size);
        std::cout << "kernel elapsed time: " << elapsed_time << "ms" << std::endl;

        std::cout << std::endl << "--------" << std::endl;

    }

}