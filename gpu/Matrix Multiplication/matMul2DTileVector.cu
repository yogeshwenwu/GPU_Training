#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}
#define WIDTH 4
#define CPU 0

// Matrix dimensions
// static const uint M = 1024;
// static const uint N = 1024;
// static const uint K = 1024;

static const uint M = 4096;
static const uint N = 4096;
static const uint K = 4096;

// Block Size for the kernel launch
static const uint BM = 128;   
static const uint BK = 128;   
static const uint BN = 8;   
static const uint TM = 8;
static const uint TK = 8;

void matMulScalar(float* matRes, float* matA, float* matB)
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

__global__ void matMul2DTileVec(float* matRes, float* matA, float* matB)
{
    __shared__ float shA[BM * BN];
    __shared__ float shB[BN * BK];

    // Global and Local index
    const uint threadRow = threadIdx.x / (BK / TK);
    const uint threadCol = threadIdx.x % (BK / TK);

    const uint globalRow = blockIdx.y * BM + threadRow * TM; // only threadIdx.x because it is 1D matrix
    const uint globalCol = blockIdx.x * BK + threadCol * TK;

    const uint localRowA = threadIdx.x / (BN / WIDTH);
    const uint localColA = threadIdx.x % (BN / WIDTH);
    const uint localRowB = threadIdx.x / (BK / WIDTH);
    const uint localColB = threadIdx.x % (BK / WIDTH);

    // const uint strideA = blockDim.x / BN;
    // const uint strideB = blockDim.x / BK; 

    float results[TM][TK] = {0.0};

    float regA[TM] = {0.0};
    float regB[TK] = {0.0};

    const uint numTiles = (N + BN -1) / BN; // 128
    for(uint tile = 0; tile < numTiles; tile++)
    {
        // load data to vector
        float4 vecA = *reinterpret_cast<float4*>(&matA[(blockIdx.y * BM + localRowA) * N + (tile * BN + localColA * WIDTH)]);
        // load vector data to SMEM
        shA[(localColA * WIDTH + 0) * BM + localRowA] = vecA.x;
        shA[(localColA * WIDTH + 1) * BM + localRowA] = vecA.y;
        shA[(localColA * WIDTH + 2) * BM + localRowA] = vecA.z;
        shA[(localColA * WIDTH + 3) * BM + localRowA] = vecA.w;

        float4 vecB = *reinterpret_cast<float4*>(&matB[(tile * BN + localRowB) * K + (blockIdx.x * BK + localColB * WIDTH)]);
        shB[localRowB * BK + localColB * WIDTH + 0] = vecB.x;
        shB[localRowB * BK + localColB * WIDTH + 1] = vecB.y;
        shB[localRowB * BK + localColB * WIDTH + 2] = vecB.z;
        shB[localRowB * BK + localColB * WIDTH + 3] = vecB.w;

        __syncthreads();

        // load row of A and col of B in registers
        for(int n = 0; n < BN; n++)
        {
            for(uint thA = 0; thA < TM; thA++)
            {
                regA[thA] = shA[(threadRow * TM + thA) + BM * n];
            }

            for(uint thB = 0; thB < TK; thB++)
            {
                regB[thB] = shB[n * BK + (threadCol * TK + thB)];
            }

            for(uint i = 0; i < TM; i++)
            {
                for(uint j = 0; j < TK; j++)
                {
                    results[i][j] += regA[i] * regB[j]; 
                }
            }
        
        }
        __syncthreads();
    }

    for(uint i = 0; i < TM; i++)
    {
        for(uint j = 0; j < TK; j++)
        {
            if(globalRow + i < M && globalCol + j < K)
            {
                matRes[(globalRow + i) * K + (globalCol + j)] = results[i][j];
            }
        }
    }   
}

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
    float* h_A     = new float[M * N];
    float* h_B     = new float[N * K];
    float* h_C_CPU = new float[M * K];
    float* h_C_GPU = new float[M * K];

    cout << "Initializing matrices (" << M << " x " << N << " x " << K << ") with random values..." << endl;

    for (int i = 0; i < M * N; i++)
        h_A[i] = static_cast<float>(rand()) / 100.0f;
    for (int i = 0; i < N * K; i++)
        h_B[i] = static_cast<float>(rand()) / 100.0f;

#if CPU
    // ========================================================
    //                     CPU BENCHMARK
    // ========================================================
    cout << "\n========== CPU Native (Scalar) Benchmark ==========\n";

    for (int i = 0; i < cpu_warmup; i++)
    {
        memset(h_C_CPU, 0, M * K * sizeof(float));
        matMulScalar(h_C_CPU, h_A, h_B);
    }

    double cpu_total_ms = 0.0;
    for (int iter = 0; iter < cpu_measure; iter++)
    {
        memset(h_C_CPU, 0, M * K * sizeof(float));   // clear before every iteration

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

    // ========================================================
    cout << "\n========== Vectorized 2D Tiled Kernel Benchmark ==========\n";

    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    dim3 blockDim(BM * BK / (TM * TK));
    dim3 gridDim((K + BK - 1) / BK, (M + BM - 1) / BM);

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
    //     matMul2DTileVec<<<gridDim, blockDim>>>(d_C, d_A, d_B);
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
        matMul2DTileVec<<<gridDim, blockDim>>>(d_C, d_A, d_B);
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