#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <iomanip>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define N (1 << 28)
#define CPU 1

using namespace std;
using namespace std::chrono;


float reductionCPU(float* iVec)
{
    double res = 0.0;
    for(int i = 0; i < N; i++)
    {
        res += iVec[i];
    }
    return res;
}

template <const uint blockSize>
__device__ void warpUnroll(volatile float * shVec, int id)
{
    if (blockSize >= 64) shVec[id] += shVec[id + 32];
    if (blockSize >= 32) shVec[id] += shVec[id + 16];
    if (blockSize >= 16) shVec[id] += shVec[id + 8];
    if (blockSize >= 8) shVec[id] += shVec[id + 4]; 
    if (blockSize >= 4) shVec[id] += shVec[id + 2]; 
    if (blockSize >= 2) shVec[id] += shVec[id + 1]; 
}

template <const uint blockSize>
__global__ void reduction5(float* iVec, float* oVec)
{
    /*
    Optimization 5: Completely Unroll the kernel  
    */

    // Thread Index
    const uint thIdx = threadIdx.x;
    const uint globalIdx = blockIdx.x * (blockDim.x * 2) + threadIdx.x; // (blockDim.x * 2) => (256 * 2)==512 => th0 (0-511) -> th1 (512-1023) 

    // Shared Memory Allocation
    extern __shared__ float shVec[];

    shVec[thIdx] = (globalIdx < N ? iVec[globalIdx] : 0.0f) + (globalIdx + blockDim.x < N ? iVec[globalIdx + blockDim.x] : 0.0f);
    __syncthreads();

    // reduction
    if(blockSize >= 1024)
    {
        if(thIdx < 512)
        {
            shVec[thIdx] += shVec[thIdx + 512];
        }
        __syncthreads();
    }
    if(blockSize >= 512)
    {
        if(thIdx < 256)
        {
            shVec[thIdx] += shVec[thIdx + 256];
        }
        __syncthreads();
    }
    if(blockSize >= 256)
    {
        if(thIdx < 128)
        {
            shVec[thIdx] += shVec[thIdx + 128];
        }
        __syncthreads();
    }
    if(blockSize >= 128)
    {
        if(thIdx < 64)
        {
            shVec[thIdx] += shVec[thIdx + 64];
        }
        __syncthreads();
    }

    if(thIdx < 32)
    {
        warpUnroll<blockSize>(shVec, thIdx);
    }

    // write back to global memory
    if(thIdx == 0)
    {
        oVec[blockIdx.x] = shVec[0];
    }
}

int main()
{
    #if CPU
    const int cpu_warmup  = 1;
    const int cpu_measure = 2;
    #endif
    const int gpu_warmup  = 5;
    const int gpu_measure = 50;

    // Init vector array
    static float h_vec[N];
    float resCPU = 0.0f;
    float resGPU = 0.0f;

    // Randomize input
    for(int i = 0; i < N; i++)
    {
        h_vec[i] = static_cast<float>(rand()) / 100.0f;
    }

    // Host code

    // CPU code
    #if CPU
        cout << "============ CPU Benchmark ============="<< endl;
        // Warm-up run
        for(int i = 0; i < cpu_warmup; i++)
        {
            resCPU = reductionCPU(h_vec);
        }

        double cpu_total_ms = 0.0;
        // Actual run
        for(int i = 0; i < cpu_measure; i++)
        {
            auto start = high_resolution_clock::now();
            resCPU = reductionCPU(h_vec);
            auto end = high_resolution_clock::now();

            duration<double, milli> elapsed = end - start;
            cpu_total_ms += elapsed.count();
            cout << "  Run " << i + 1 << ": " << fixed << setprecision(4) << elapsed.count() << " ms\n";
        }
        double cpu_avg_ms = cpu_total_ms / cpu_measure;

        cout << "---------------------------------------------------\n";
        cout << "CPU Average time     : " << fixed << setprecision(4) << cpu_avg_ms << " ms\n" << endl;
    #endif

    cout << "========== GPU reduction5 (Sequential) Benchmark ==========" << endl;
    // Kernel Launch Configuration
    const uint threadsList[6] = {32, 64, 128, 256, 512, 1024};
    const uint blockSize = threadsList[3]; // 256
    const uint gridSize = (N + (2* blockSize) -1) / (2 * blockSize); // 

    float d_res[gridSize] = {0.0f};
    float* d_vec;
    float* d_resVec;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float total_h2d = 0.0f, total_kernel = 0.0f, total_d2h = 0.0f;

    CUDA_CHECK(cudaMalloc(&d_vec, N * sizeof(float))); 
    CUDA_CHECK(cudaMalloc(&d_resVec, gridSize * sizeof(float)));

    for (int i = 0; i < gpu_warmup; i++)
    {
        // HTD copy
        CUDA_CHECK(cudaMemcpy(d_vec, h_vec, N * sizeof(float), cudaMemcpyHostToDevice));

        // Launch Kernel
        switch(blockSize)
        {
            case 1024:
                reduction5<1024><<<gridSize, 1024, 1024 * sizeof(float)>>>(d_vec, d_resVec);
                break;
            case 512:
                reduction5<512><<<gridSize, 512, 512 * sizeof(float)>>>(d_vec, d_resVec);
                break;
            case 256:
                reduction5<256><<<gridSize, 256, 256 * sizeof(float)>>>(d_vec, d_resVec);
                break;
            case 128:
                reduction5<128><<<gridSize, 128, 128 * sizeof(float)>>>(d_vec, d_resVec);
                break;
        }
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // DTH copy
        CUDA_CHECK(cudaMemcpy(d_res, d_resVec, gridSize * sizeof(float), cudaMemcpyDeviceToHost)); // return partial sum, gridSize no of elements are returned. 
    }

    for (int i = 0; i < gpu_measure; i++)
    {
        // HTD copy
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_vec, h_vec, N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float h2d_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));
        total_h2d += h2d_ms;

        // Launch Kernel
        CUDA_CHECK(cudaEventRecord(start));
        switch(blockSize)
        {
            case 1024:
                reduction5<1024><<<gridSize, 1024, 1024 * sizeof(float)>>>(d_vec, d_resVec);
                break;
            case 512:
                reduction5<512><<<gridSize, 512, 512 * sizeof(float)>>>(d_vec, d_resVec);
                break;
            case 256:
                reduction5<256><<<gridSize, 256, 256 * sizeof(float)>>>(d_vec, d_resVec);
                break;
            case 128:
                reduction5<128><<<gridSize, 128, 128 * sizeof(float)>>>(d_vec, d_resVec);
                break;
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
        total_kernel += kernel_ms;

        // DTH copy
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_res, d_resVec, gridSize * sizeof(float), cudaMemcpyDeviceToHost)); // return partial sum, gridSize no of elements are returned. 
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float d2h_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));
        total_d2h += d2h_ms;

        cout << "  Run " << setw(2) << i + 1 << " | H2D: " << fixed << setprecision(4) << h2d_ms
             << " ms | Kernel: " << kernel_ms << " ms | D2H: " << d2h_ms << " ms\n";
    }

    // Final result
    for(uint i = 0; i < gridSize; i++)
    {
        resGPU += d_res[i];
    } 

    float avg_h2d    = total_h2d    / gpu_measure;
    float avg_kernel = total_kernel / gpu_measure;
    float avg_d2h    = total_d2h    / gpu_measure;
    float avg_total  = avg_h2d + avg_kernel + avg_d2h;

    // -------------------- Final Report --------------------
    cout << "\n---------------------------------------------------\n";
    cout << "Average H2D time       : " << fixed << setprecision(4) << avg_h2d    << " ms\n";
    cout << "Average Kernel time    : " << fixed << setprecision(4) << avg_kernel << " ms\n";
    cout << "Average D2H time       : " << fixed << setprecision(4) << avg_d2h    << " ms\n";
    cout << "Average Total GPU time : " << fixed << setprecision(4) << avg_total  << " ms\n";
    cout << "---------------------------------------------------\n";

    // validate
    #if CPU 
        float diff = fabs(resCPU - resGPU) / fmax(fabs(resCPU), fabs(resGPU));
        if (diff > 1e-3f)
        {
            cout << "Mismatch in result::  CPU: " << resCPU << " & GPU: " << resGPU << endl;
        }
        else
        {
            cout << "Results Passed" << endl;
        }
    #endif
}