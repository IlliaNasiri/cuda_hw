#include "utils.h"

__global__ void matmul_thread_per_row(
    const float* A, // dimensions: [M, K]
    const float* B, // dimensions: [K, N]
    float* C, // dimensions: [M, N]
    const unsigned int M,
    const unsigned int K,
    const unsigned int N
) {
    // we have n_cols values to calculate
    unsigned int thread_row = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_row < M) {
        for (unsigned int i = 0; i < N; i++) {
            float sum = 0.0f;
            for (unsigned int j = 0; j < K; j++)
                // A[thread_row][j] * B[j][i] , j is the intermediate dimension over which we walk.
                    // we do this N times to populate each entry in the row using the outer loop.
                        sum += A[thread_row * K + j] * B[j * N + i];

            C[thread_row * N + i] = sum;

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

    // Timing done based on the following tutorial https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_thread_per_row<<< grid_size, block_size >>>(A.d_data, B.d_data, C.d_data, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    C.to_host();
    print_matrix_hw2(C);

    return milliseconds;
}

int main() {
    unsigned int M = 20000;
    unsigned int K = 20000;
    unsigned int N = 20000;

    dim3 block_sizes[] = {
        dim3(32, 1, 1),
        dim3(16, 1, 1),
        dim3(8, 1, 1)
    };

    for (auto block_size : block_sizes) {
        //
        float block_x = block_size.x;
        dim3 grid_size( ceil(M / block_x), 1, 1);

        std::cout << "block size: " << dim3_to_string(block_size) << " , grid_size" << dim3_to_string(grid_size) << std::endl;

        float elapsed_time = block_size_experiments(M, K, N, grid_size, block_size);
        std::cout << "kernel elapsed time: " << elapsed_time << "ms" << std::endl;

        std::cout << std::endl << "--------" << std::endl;

    }
}