#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
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

void vecAddNative(vector<float>& vecRes, const vector<float>& vecA, const vector<float>& vecB)
{
    for (int i = 0; i < N; i++)
    {
        vecRes[i] = vecA[i] + vecB[i];
    }
}

bool validateCPU(const vector<float>& cpu, const vector<float>& gpu, float tol = 1e-5f)
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

    cout << "Initializing vectors (N = " << N << ")..." << endl;

    vector<float> arr(N);
    vector<float> brr(N);
    vector<float> resCPU(N);
    vector<float> resGPU(N);

    for (int i = 0; i < N; i++)
    {
        arr[i] = static_cast<float>(i % 100);
        brr[i] = static_cast<float>((i * 3) % 100);
    }

#if CPU
    cout << "\n========== CPU Native (Scalar) Benchmark ==========\n";

    for (int i = 0; i < cpu_warmup; i++)
    {
        vecAddNative(resCPU, arr, brr);
    }

    double cpu_total_ms = 0.0;

    for (int iter = 0; iter < cpu_measure; iter++)
    {
        auto start = high_resolution_clock::now();
        vecAddNative(resCPU, arr, brr);
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

    cout << "\n========== CUDA Vector Add Benchmark ==========\n";

    size_t size = N * sizeof(float);

    float *devA = nullptr, *devB = nullptr, *devC = nullptr;
    cudaMalloc(&devA, size);
    cudaMalloc(&devB, size);
    cudaMalloc(&devC, size);

    int threads = BLOCK_SIZE;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < gpu_warmup; i++)
    {
        cudaMemcpy(devA, arr.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(devB, brr.data(), size, cudaMemcpyHostToDevice);
        cudaMemset(devC, 0, size);
        vecAddKernel<<<blocks, threads>>>(devC, devA, devB, N);
        cudaMemcpy(resGPU.data(), devC, size, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    float total_h2d = 0.0f;
    float total_kernel = 0.0f;
    float total_d2h = 0.0f;

    for (int iter = 0; iter < gpu_measure; iter++)
    {
        cudaEventRecord(start);
        cudaMemcpy(devA, arr.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(devB, brr.data(), size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float h2d_ms = 0.0f;
        cudaEventElapsedTime(&h2d_ms, start, stop);
        total_h2d += h2d_ms;

        cudaMemset(devC, 0, size);
        cudaEventRecord(start);
        vecAddKernel<<<blocks, threads>>>(devC, devA, devB, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, start, stop);
        total_kernel += kernel_ms;

        cudaEventRecord(start);
        cudaMemcpy(resGPU.data(), devC, size, cudaMemcpyDeviceToHost);
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

    float avg_h2d = total_h2d / gpu_measure;
    float avg_kernel = total_kernel / gpu_measure;
    float avg_d2h = total_d2h / gpu_measure;
    float avg_total = avg_h2d + avg_kernel + avg_d2h;

    double gpu_gflops = (ops / (avg_kernel * 1e-3)) / 1e9;
    double gpu_bandwidth = (bytes / (avg_kernel * 1e-3)) / 1e9;

#if CPU
    cout << "\nValidating GPU result against CPU..." << endl;
    validateCPU(resCPU, resGPU);
#endif

    cout << "\n---------------------------------------------------\n";
    cout << "Average H2D time     : " << fixed << setprecision(3) << avg_h2d << " ms\n";
    cout << "Average Kernel time  : " << fixed << setprecision(3) << avg_kernel << " ms\n";
    cout << "Average D2H time     : " << fixed << setprecision(3) << avg_d2h << " ms\n";
    cout << "Average Total time   : " << fixed << setprecision(3) << avg_total << " ms\n";
    cout << "---------------------------------------------------\n";
    cout << "Pure Kernel GFLOPS   : " << fixed << setprecision(2) << gpu_gflops << " GFLOPS\n";
    cout << "Pure Kernel Bandwidth: " << fixed << setprecision(2) << gpu_bandwidth << " GB/s\n";
#if CPU
    cout << "Speedup (Kernel vs CPU): " << fixed << setprecision(2)
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
