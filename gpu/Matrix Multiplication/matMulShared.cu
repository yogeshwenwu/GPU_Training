#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>          // for rand()

using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE 32
#define CPU 0

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// static const uint M = 1024;
// static const uint N = 1024;
// static const uint K = 1024;
static const uint M = 4096;
static const uint N = 4096;
static const uint K = 4096;

// ====================== CPU Native Kernel ======================
void matMulScalar(float* matRes, float* matA, float* matB)
{
    for (int m = 0; m < M; m++)
    {
        for (int k = 0; k < K; k++)
        {
            float sum = 0.0f;
            for (int n = 0; n < N; n++)
            {
                sum += matA[m * N + n] * matB[n * K + k];
            }
            matRes[m * K + k] = sum;
        }
    }
}

// ====================== Shared Memory Tiled Kernel ======================
__global__ void matMulShared(float* matRes, float* matA, float* matB)
{
    __shared__ float shA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float shB[BLOCK_SIZE * BLOCK_SIZE];

    const uint row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const uint col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;
    const int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < numTiles; tile++)
    {
        const int aCol = tile * BLOCK_SIZE + threadIdx.x;
        const int bRow = tile * BLOCK_SIZE + threadIdx.y;

        // Load A and B into shared memory (with bounds checking)
        shA[threadIdx.y * BLOCK_SIZE + threadIdx.x] = (row < M && aCol < N) ? matA[row * N + aCol] : 0.0f;
        shB[threadIdx.y * BLOCK_SIZE + threadIdx.x] = (bRow < N && col < K) ? matB[bRow * K + col] : 0.0f;

        __syncthreads();

        for (int n = 0; n < BLOCK_SIZE; n++)
        {
            sum += shA[threadIdx.y * BLOCK_SIZE + n] * shB[n * BLOCK_SIZE + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K)
    {
        matRes[row * K + col] = sum;
    }
}

// ====================== CPU Validation ======================
bool validateCPU(const float* cpu, const float* gpu, float tol = 1e-3f)
{
    int mismatches = 0;
    const int max_print = 5;

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            float a = cpu[i * K + j];
            float b = gpu[i * K + j];
            float diff = fabs(a - b) / fmax(fabs(a), fabs(b));

            if (diff > tol)
            {
                if (mismatches < max_print)
                {
                    printf("Mismatch at (%d, %d): CPU = %f, GPU = %f, rel_diff = %f\n",
                           i, j, a, b, diff);
                }
                mismatches++;
            }
        }
    }

    if (mismatches == 0)
    {
        cout << "Validation passed: All values match within tolerance (" << tol << ")" << endl;
        return true;
    }
    else
    {
        cout << "Validation failed: " << mismatches << " mismatches found." << endl;
        return false;
    }
}

int main()
{
    #if CPU
    const int cpu_warmup  = 1;
    const int cpu_measure = 2;
    #endif
    // const int gpu_warmup  = 5;
    const int gpu_measure = 1;

    // -------------------- Host memory --------------------
    float* h_A      = new float[M * N];
    float* h_B      = new float[N * K];
    float* h_C_CPU  = new float[M * K];
    float* h_C_GPU  = new float[M * K];

    cout << "Initializing matrices (" << M << " x " << N << " x " << K << ") with random values..." << endl;

    // Random initialization (same as your code)
    for (int i = 0; i < M * N; i++)
        h_A[i] = static_cast<float>(rand()) / 100.0f;
    for (int i = 0; i < N * K; i++)
        h_B[i] = static_cast<float>(rand()) / 100.0f;

    #if CPU
    cout << "\n========== CPU Native (Scalar) Benchmark ==========\n";

    // Warm-up
    for (int i = 0; i < cpu_warmup; i++)
    {
        memset(h_C_CPU, 0, M * K * sizeof(float));
        matMulScalar(h_C_CPU, h_A, h_B);
    }

    double cpu_total_ms = 0.0;
    for (int iter = 0; iter < cpu_measure; iter++)
    {
        memset(h_C_CPU, 0, M * K * sizeof(float));          // clear before every iteration

        auto start = high_resolution_clock::now();
        matMulScalar(h_C_CPU, h_A, h_B);
        auto end = high_resolution_clock::now();

        duration<double, milli> elapsed = end - start;
        cpu_total_ms += elapsed.count();
        cout << "  Run " << iter + 1 << ": " << fixed << setprecision(2) << elapsed.count() << " ms\n";
    }

    double cpu_avg_ms = cpu_total_ms / cpu_measure;
    #endif
    double flops = 2.0 * static_cast<double>(M) * N * K;
    
    #if CPU
    double cpu_gflops = (flops / (cpu_avg_ms * 1e-3)) / 1e9;

    cout << "---------------------------------------------------\n";
    cout << "CPU Average time     : " << fixed << setprecision(2) << cpu_avg_ms << " ms\n";
    cout << "CPU Performance      : " << fixed << setprecision(2) << cpu_gflops << " GFLOPS\n";
    cout << "===================================================\n";
    #endif

    cout << "\n========== Shared Memory Tiled Kernel Benchmark ==========\n";

    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((K + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // // ----- Warm-up -----
    // cout << "Warm-up..." << endl;
    // for (int i = 0; i < gpu_warmup; i++)
    // {
    //     CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    //     CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    //     CUDA_CHECK(cudaMemset(d_C, 0, sizeC));
    //     matMulShared<<<gridDim, blockDim>>>(d_C, d_A, d_B);
    //     CUDA_CHECK(cudaGetLastError());
    //     CUDA_CHECK(cudaMemcpy(h_C_GPU, d_C, sizeC, cudaMemcpyDeviceToHost));
    // }
    // CUDA_CHECK(cudaDeviceSynchronize());

    // ----- Measured runs -----
    float total_h2d = 0.0f, total_kernel = 0.0f, total_d2h = 0.0f;

    for (int iter = 0; iter < gpu_measure; iter++)
    {
        // H2D
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float h2d_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));
        total_h2d += h2d_ms;

        // Kernel (clear before every launch)
        CUDA_CHECK(cudaMemset(d_C, 0, sizeC));
        CUDA_CHECK(cudaEventRecord(start));
        matMulShared<<<gridDim, blockDim>>>(d_C, d_A, d_B);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
        total_kernel += kernel_ms;

        // D2H
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(h_C_GPU, d_C, sizeC, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float d2h_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));
        total_d2h += d2h_ms;

        cout << "  Run " << setw(2) << iter + 1
             << " | H2D: " << fixed << setprecision(3) << h2d_ms
             << " ms | Kernel: " << kernel_ms
             << " ms | D2H: " << d2h_ms << " ms\n";
    }

    float avg_h2d    = total_h2d    / gpu_measure;
    float avg_kernel = total_kernel / gpu_measure;
    float avg_d2h    = total_d2h    / gpu_measure;
    float avg_total  = avg_h2d + avg_kernel + avg_d2h;

    double gpu_gflops = (flops / (avg_kernel * 1e-3)) / 1e9;

    #if CPU
    // -------------------- Validation --------------------
    cout << "\nValidating GPU result against CPU..." << endl;
    validateCPU(h_C_CPU, h_C_GPU);
    #endif

    // -------------------- Final Report --------------------
    cout << "\n---------------------------------------------------\n";
    cout << "Average H2D time       : " << fixed << setprecision(3) << avg_h2d    << " ms\n";
    cout << "Average Kernel time    : " << fixed << setprecision(3) << avg_kernel << " ms\n";
    cout << "Average D2H time       : " << fixed << setprecision(3) << avg_d2h    << " ms\n";
    cout << "Average Total GPU time : " << fixed << setprecision(3) << avg_total  << " ms\n";
    cout << "---------------------------------------------------\n";
    cout << "Pure Kernel GFLOPS     : " << fixed << setprecision(2) << gpu_gflops << " GFLOPS\n";
    #if CPU
    cout << "Speedup (Kernel vs CPU): " << fixed << setprecision(2)
         << (cpu_avg_ms / avg_kernel) << "x\n";
    #endif
    cout << "===================================================\n";

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_CPU;
    delete[] h_C_GPU;

    cout << "\nBenchmark complete." << endl;
    return 0;
}
