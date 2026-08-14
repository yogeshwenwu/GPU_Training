#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

#define DEBUG 0
#define BLOCK_SIZE 32
#define CPU 0

// static const uint M = 1024;
// static const uint N = 1024;
// static const uint K = 1024;

static const uint M = 4096;
static const uint N = 4096;
static const uint K = 4096;

using namespace std;
using namespace std::chrono;

// CPU Native Kernel
void MatMulScalar(float* matRes, float* matA, float* matB)
{
    for (int m = 0; m < M; m++)
    {
        for (int k = 0; k < K; k++)
        {
            float sum = 0.0f;
            for (int n = 0; n < N; n++)
            {
                sum = sum + (matA[m * N + n] * matB[n * K + k]);
            }
            matRes[m * K + k] = sum;
        }
    }
}

// GPU Native Kernel
__global__ void MatMulKernel(float* matRes, float* matA, float* matB)
{
    /*
    - Launch Matrix C Dimension (M * K) number of threads.
    - Each thread computes one element of the result matrix C, which is the dot product of one row of A and one column of B.
    */
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 35 => 0 * 32 + 1 = 1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 35 => 0 * 32 + 3  = 3

    if (row < M && col < K)
    {
        float sum = 0.0f;
        for (int n = 0; n < N; n++)
        {
            sum += matA[row * N + n] * matB[n * K + col];
        }
        matRes[row * K + col] = sum;
    }
}

bool validateCPU(const float* cpu, const float* gpu, float tol = 1e-3f)
{
    int mismatches = 0;
    const int max_print = 10;   // print only first few mismatches

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
    // const int gpu_warmup   = 5;
    const int gpu_measure = 1;

    // Init matrices
    static float matrixA[M][N];
    static float matrixB[N][K];
    static float matrixResCPU[M][K];
    static float matrixResGPU[M][K];

    cout << "Initializing matrices (" << M << "x" << N << "x" << K << ")..." << endl;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            matrixA[i][j] = static_cast<float>(N * i + j);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++)
            matrixB[i][j] = static_cast<float>(i + j);

    #if CPU
    cout << "\n========== CPU Native (Scalar) Benchmark ==========\n";

    for (int i = 0; i < cpu_warmup; i++)
    {
        memset(matrixResCPU, 0, sizeof(matrixResCPU));
        MatMulScalar((float*)matrixResCPU, (float*)matrixA, (float*)matrixB);
    }
    
    

    double cpu_total_ms = 0.0;

    for (int iter = 0; iter < cpu_measure; iter++)
    {
        auto start = high_resolution_clock::now();
        MatMulScalar((float*)matrixResCPU, (float*)matrixA, (float*)matrixB);
        auto end = high_resolution_clock::now();

        duration<double, milli> elapsed = end - start;
        cpu_total_ms += elapsed.count();

        cout << "  Run " << iter + 1 << ": "
             << fixed << setprecision(2) << elapsed.count() << " ms" << endl;
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

    cout << "\n========== Naive CUDA Kernel Benchmark ==========\n";

    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // CUDA Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // // Warmup run
    // cout << "Warm-up..." << endl;
    // for (int i = 0; i < gpu_warmup; i++)
    // {
    //     cudaMemcpy(d_A, matrixA, sizeA, cudaMemcpyHostToDevice);
    //     cudaMemcpy(d_B, matrixB, sizeB, cudaMemcpyHostToDevice);
    //     cudaMemset(d_C, 0, sizeC);
    //     MatMulKernel<<<gridDim, blockDim>>>(d_C, d_A, d_B);
    //     cudaMemcpy(matrixResGPU, d_C, sizeC, cudaMemcpyDeviceToHost);
    // }
    // cudaDeviceSynchronize();

    // Actual run
    float total_h2d = 0.0f;
    float total_kernel = 0.0f;
    float total_d2h = 0.0f;

    for (int iter = 0; iter < gpu_measure; iter++)
    {
        // H2D 
        cudaEventRecord(start);
        cudaMemcpy(d_A, matrixA, sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, matrixB, sizeB, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float h2d_ms = 0.0f;
        cudaEventElapsedTime(&h2d_ms, start, stop);
        total_h2d += h2d_ms;

        // Launch Kernel 
        cudaMemset(d_C, 0, sizeC);
        cudaEventRecord(start);
        MatMulKernel<<<gridDim, blockDim>>>(d_C, d_A, d_B);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, start, stop);
        total_kernel += kernel_ms;

        // D2H
        cudaEventRecord(start);
        cudaMemcpy(matrixResGPU, d_C, sizeC, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float d2h_ms = 0.0f;
        cudaEventElapsedTime(&d2h_ms, start, stop);
        total_d2h += d2h_ms;

        cout << "  Run " << iter + 1 
             << " | H2D: " << fixed << setprecision(3) << h2d_ms
             << " ms | Kernel: " << kernel_ms
             << " ms | D2H: " << d2h_ms << " ms" << endl;
    }

    float avg_h2d    = total_h2d    / gpu_measure;
    float avg_kernel = total_kernel / gpu_measure;
    float avg_d2h    = total_d2h    / gpu_measure;
    float avg_total  = avg_h2d + avg_kernel + avg_d2h;

    double gpu_gflops = (flops / (avg_kernel * 1e-3)) / 1e9;   // pure compute GFLOPS

    #if CPU
    // Validate 
    cout << "\nValidating GPU result against CPU..." << endl;
    validateCPU((float*)matrixResCPU, (float*)matrixResGPU);
    #endif

    // Benchmark Results
    cout << "\n---------------------------------------------------\n";
    cout << "Average H2D time     : " << fixed << setprecision(3) << avg_h2d    << " ms\n";
    cout << "Average Kernel time  : " << fixed << setprecision(3) << avg_kernel << " ms\n";
    cout << "Average D2H time     : " << fixed << setprecision(3) << avg_d2h    << " ms\n";
    cout << "Average Total time   : " << fixed << setprecision(3) << avg_total  << " ms\n";
    cout << "---------------------------------------------------\n";
    cout << "Pure Kernel GFLOPS   : " << fixed << setprecision(2) << gpu_gflops << " GFLOPS\n";
    #if  CPU
    cout << "Speedup (Kernel vs CPU): " << fixed << setprecision(2)
         << (cpu_avg_ms / avg_kernel) << "x\n";
    #endif
    cout << "===================================================\n";

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cout << "\nBenchmark complete." << endl;
    return 0;
}
