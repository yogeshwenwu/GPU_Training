#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#define DEBUG 0
#define BLOCK_SIZE 256
#define CPU 1

static const int N = 1 << 26;

using namespace std;
using namespace std::chrono;

__global__ void vecAddKernel(float* vecRes, const float* vecA, const float* vecB, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n)
    {
        vecRes[i] = vecA[i] + vecB[i];
    }
}

void vecAddNative(vector<float>& vecRes, const float* vecA, const float* vecB)
{
    for (int i = 0; i < N; i++)
    {
        vecRes[i] = vecA[i] + vecB[i];
    }
}

bool validateCPU(const vector<float>& cpu, const float* gpu, float tol = 1e-5f)
{
    int mismatches = 0;
    const int max_print = 10;

    for (int i = 0; i < N; i++)
    {
        float diff = fabs(cpu[i] - gpu[i]);

        if (diff > tol)
        {
            if (mismatches < max_print)
            {
                cout << "Mismatch at index " << i
                     << ": CPU = " << cpu[i]
                     << ", GPU = " << gpu[i]
                     << ", abs_diff = " << diff << endl;
            }
            mismatches++;
        }
    }

    if (mismatches == 0)
    {
        cout << "Validation passed: All values match within tolerance (" << tol << ")" << endl;
        return true;
    }

    cout << "Validation failed: " << mismatches << " mismatches found." << endl;
    return false;
}

int main()
{
#if CPU
    const int cpu_warmup = 1;
    const int cpu_measure = 3;
#endif
    const int gpu_warmup = 3;
    const int gpu_measure = 10;

    cout << "Initializing managed vectors (N = " << N << ")..." << endl;

    size_t size = N * sizeof(float);
    vector<float> resCPU(N);

    float *devA = nullptr, *devB = nullptr, *devC = nullptr;
    cudaMallocManaged(&devA, size);
    cudaMallocManaged(&devB, size);
    cudaMallocManaged(&devC, size);

    for (int i = 0; i < N; i++)
    {
        devA[i] = static_cast<float>(i % 100);
        devB[i] = static_cast<float>((i * 3) % 100);
        devC[i] = 0.0f;
    }

#if CPU
    cout << "\n========== CPU Native (Scalar) Benchmark ==========\n";

    for (int i = 0; i < cpu_warmup; i++)
    {
        vecAddNative(resCPU, devA, devB);
    }

    double cpu_total_ms = 0.0;

    for (int iter = 0; iter < cpu_measure; iter++)
    {
        auto start = high_resolution_clock::now();
        vecAddNative(resCPU, devA, devB);
        auto end = high_resolution_clock::now();

        duration<double, milli> elapsed = end - start;
        cpu_total_ms += elapsed.count();

        cout << "  Run " << iter + 1 << ": "
             << fixed << setprecision(3) << elapsed.count() << " ms" << endl;
    }

    double cpu_avg_ms = cpu_total_ms / cpu_measure;
#endif

    double ops = static_cast<double>(N);
    double bytes = 3.0 * static_cast<double>(N) * sizeof(float);

#if CPU
    double cpu_gflops = (ops / (cpu_avg_ms * 1e-3)) / 1e9;
    double cpu_bandwidth = (bytes / (cpu_avg_ms * 1e-3)) / 1e9;

    cout << "---------------------------------------------------\n";
    cout << "CPU Average time     : " << fixed << setprecision(3) << cpu_avg_ms << " ms\n";
    cout << "CPU Performance      : " << fixed << setprecision(2) << cpu_gflops << " GFLOPS\n";
    cout << "CPU Bandwidth        : " << fixed << setprecision(2) << cpu_bandwidth << " GB/s\n";
    cout << "===================================================\n";
#endif

    cout << "\n========== CUDA Managed Memory Vector Add Benchmark ==========\n";

    int threads = BLOCK_SIZE;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < gpu_warmup; i++)
    {
        cudaMemset(devC, 0, size);
        vecAddKernel<<<blocks, threads>>>(devC, devA, devB, N);
    }
    cudaDeviceSynchronize();

    float total_kernel = 0.0f;

    for (int iter = 0; iter < gpu_measure; iter++)
    {
        cudaMemset(devC, 0, size);
        cudaEventRecord(start);
        vecAddKernel<<<blocks, threads>>>(devC, devA, devB, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, start, stop);
        total_kernel += kernel_ms;

        cout << "  Run " << iter + 1
             << " | Kernel: " << fixed << setprecision(3) << kernel_ms << " ms" << endl;
    }

    float avg_kernel = total_kernel / gpu_measure;
    float avg_total = avg_kernel;

    double gpu_gflops = (ops / (avg_kernel * 1e-3)) / 1e9;
    double gpu_bandwidth = (bytes / (avg_kernel * 1e-3)) / 1e9;

#if CPU
    cout << "\nValidating GPU result against CPU..." << endl;
    validateCPU(resCPU, devC);
#endif

    cout << "\n---------------------------------------------------\n";
    cout << "Average Kernel time      : " << fixed << setprecision(3) << avg_kernel << " ms\n";
    cout << "Average Total time       : " << fixed << setprecision(3) << avg_total << " ms\n";
    cout << "---------------------------------------------------\n";
    cout << "Pure Kernel GFLOPS       : " << fixed << setprecision(2) << gpu_gflops << " GFLOPS\n";
    cout << "Pure Kernel Bandwidth    : " << fixed << setprecision(2) << gpu_bandwidth << " GB/s\n";
#if CPU
    cout << "Speedup (Kernel vs CPU)  : " << fixed << setprecision(2)
         << (cpu_avg_ms / avg_kernel) << "x\n";
#endif
    cout << "===================================================\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    cout << "\nBenchmark complete." << endl;
    return 0;
}
