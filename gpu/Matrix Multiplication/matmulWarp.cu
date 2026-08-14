#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <cmath>

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
#define WARPSIZE 32
#define CPU 0   // set to 1 for CPU reference + validation

// Matrix dimensions (use 1024 for debugging, later switch to 4096)
// static const uint M = 1024;
// static const uint N = 1024;
// static const uint K = 1024;

static const uint M = 4096;
static const uint N = 4096;
static const uint K = 4096;

// ------------------------------------------------------------------
// Tiling parameters (CORRECTED hierarchy)
// ------------------------------------------------------------------
static const uint BM = 128;
static const uint BK = 128;
static const uint BN = 16;

static const uint WM = 64;
static const uint WK = 64;

static const uint WMITER = 1;
static const uint WKITER = 2;
static const uint WSUBM  = WM / WMITER;   // 64
static const uint WSUBK  = WK / WKITER;   // 32

static const uint TM = 8;
static const uint TK = 8;

// (WSUBM/TM) * (WSUBK/TK) = 8 * 4 = 32  → matches warp size
static const uint NUM_THREADS = (BM / WM) * (BK / WK) * WARPSIZE; // 128

// ------------------------------------------------------------------
// CPU reference
// ------------------------------------------------------------------
void matMulScalar(float* matRes, float* matA, float* matB)
{
    for (int m = 0; m < (int)M; m++) {
        for (int k = 0; k < (int)K; k++) {
            float sum = 0.0f;
            for (int n = 0; n < (int)N; n++) {
                sum += matA[m * N + n] * matB[n * K + k];
            }
            matRes[m * K + k] = sum;
        }
    }
}

// ------------------------------------------------------------------
// Warp-tiled kernel (fully fixed)
// ------------------------------------------------------------------
__global__ void matMulWarpTiled(float* matRes, float* matA, float* matB)
{
    __shared__ float shA[BM * BN];
    __shared__ float shB[BN * BK];

    // Warp placement inside the block tile
    const uint warpIdx = threadIdx.x / WARPSIZE;
    const uint warpCol = warpIdx % (BK / WK);
    const uint warpRow = warpIdx / (BK / WK);

    // Thread placement inside the warp sub-tile
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    const uint threadColInWarp = threadIdxInWarp % (WSUBK / TK); // 0..3
    const uint threadRowInWarp = threadIdxInWarp / (WSUBK / TK); // 0..7

    const uint globalRow = blockIdx.y * BM + warpRow * WM + threadRowInWarp * TM;
    const uint globalCol = blockIdx.x * BK + warpCol * WK + threadColInWarp * TK;

    // Loading indices (multi-step)
    const uint innerRowA = threadIdx.x / (BN / WIDTH);
    const uint innerColA = threadIdx.x % (BN / WIDTH);
    const uint rowStrideA = NUM_THREADS / (BN / WIDTH); // 32

    const uint innerRowB = threadIdx.x / (BK / WIDTH);
    const uint innerColB = threadIdx.x % (BK / WIDTH);
    const uint rowStrideB = NUM_THREADS / (BK / WIDTH); // 4

    float results[WMITER * TM][WKITER * TK] = {{0.0f}};
    float regA[WMITER * TM];
    float regB[WKITER * TK];

    const uint numTiles = (N + BN - 1) / BN;

    for (uint tile = 0; tile < numTiles; ++tile)
    {
        // ---------- Load A (transposed into shared memory) ----------
        for (uint offset = 0; offset < BM; offset += rowStrideA) {
            uint row = blockIdx.y * BM + innerRowA + offset;
            uint col = tile * BN + innerColA * WIDTH;

            float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};
            if (row < M && col + 3 < N) {
                tmp = *reinterpret_cast<const float4*>(&matA[row * N + col]);
            } else if (row < M) {
                if (col + 0 < N) tmp.x = matA[row * N + col + 0];
                if (col + 1 < N) tmp.y = matA[row * N + col + 1];
                if (col + 2 < N) tmp.z = matA[row * N + col + 2];
                if (col + 3 < N) tmp.w = matA[row * N + col + 3];
            }

            shA[(innerColA * WIDTH + 0) * BM + (innerRowA + offset)] = tmp.x;
            shA[(innerColA * WIDTH + 1) * BM + (innerRowA + offset)] = tmp.y;
            shA[(innerColA * WIDTH + 2) * BM + (innerRowA + offset)] = tmp.z;
            shA[(innerColA * WIDTH + 3) * BM + (innerRowA + offset)] = tmp.w;
        }

        // ---------- Load B ----------
        for (uint offset = 0; offset < BN; offset += rowStrideB) {
            uint row = tile * BN + innerRowB + offset;
            uint col = blockIdx.x * BK + innerColB * WIDTH;

            float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};
            if (row < N && col + 3 < K) {
                tmp = *reinterpret_cast<const float4*>(&matB[row * K + col]);
            } else if (row < N) {
                if (col + 0 < K) tmp.x = matB[row * K + col + 0];
                if (col + 1 < K) tmp.y = matB[row * K + col + 1];
                if (col + 2 < K) tmp.z = matB[row * K + col + 2];
                if (col + 3 < K) tmp.w = matB[row * K + col + 3];
            }

            shB[(innerRowB + offset) * BK + innerColB * WIDTH + 0] = tmp.x;
            shB[(innerRowB + offset) * BK + innerColB * WIDTH + 1] = tmp.y;
            shB[(innerRowB + offset) * BK + innerColB * WIDTH + 2] = tmp.z;
            shB[(innerRowB + offset) * BK + innerColB * WIDTH + 3] = tmp.w;
        }

        __syncthreads();

        // ---------- Compute ----------
        for (uint n = 0; n < BN; ++n)
        {
            #pragma unroll
            for (uint wSubRow = 0; wSubRow < WMITER; ++wSubRow) {
                #pragma unroll
                for (uint i = 0; i < TM; ++i) {
                    regA[wSubRow * TM + i] =
                        shA[n * BM + (warpRow * WM + wSubRow * WSUBM + threadRowInWarp * TM + i)];
                }
            }

            #pragma unroll
            for (uint wSubCol = 0; wSubCol < WKITER; ++wSubCol) {
                #pragma unroll
                for (uint j = 0; j < TK; ++j) {
                    regB[wSubCol * TK + j] =
                        shB[n * BK + (warpCol * WK + wSubCol * WSUBK + threadColInWarp * TK + j)];
                }
            }

            #pragma unroll
            for (uint wSubRow = 0; wSubRow < WMITER; ++wSubRow) {
                #pragma unroll
                for (uint wSubCol = 0; wSubCol < WKITER; ++wSubCol) {
                    #pragma unroll
                    for (uint i = 0; i < TM; ++i) {
                        #pragma unroll
                        for (uint j = 0; j < TK; ++j) {
                            results[wSubRow * TM + i][wSubCol * TK + j] +=
                                regA[wSubRow * TM + i] * regB[wSubCol * TK + j];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---------- Store results ----------
    #pragma unroll
    for (uint wSubRow = 0; wSubRow < WMITER; ++wSubRow) {
        #pragma unroll
        for (uint wSubCol = 0; wSubCol < WKITER; ++wSubCol) {
            const uint row = globalRow + wSubRow * WSUBM;
            const uint col = globalCol + wSubCol * WSUBK;

            #pragma unroll
            for (uint i = 0; i < TM; ++i) {
                #pragma unroll
                for (uint j = 0; j < TK; ++j) {
                    if (row + i < M && col + j < K) {
                        matRes[(row + i) * K + (col + j)] =
                            results[wSubRow * TM + i][wSubCol * TK + j];
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// Validation helper
// ------------------------------------------------------------------
bool validateCPU(const float* cpu, const float* gpu, float tol = 1e-3f)
{
    int mismatches = 0;
    const int max_print = 5;

    for (int i = 0; i < (int)M; i++) {
        for (int j = 0; j < (int)K; j++) {
            float a = cpu[i * K + j];
            float b = gpu[i * K + j];
            float denom = fmax(fabs(a), fabs(b));
            float diff = (denom > 1e-12f) ? fabs(a - b) / denom : fabs(a - b);

            if (diff > tol) {
                if (mismatches < max_print) {
                    printf("Mismatch at (%d, %d): CPU = %f, GPU = %f, rel_diff = %f\n",
                           i, j, a, b, diff);
                }
                mismatches++;
            }
        }
    }

    if (mismatches == 0) {
        cout << "Validation passed: All values match within tolerance (" << tol << ")" << endl;
        return true;
    } else {
        cout << "Validation failed: " << mismatches << " mismatches found." << endl;
        return false;
    }
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
int main()
{
#if CPU
    const int cpu_warmup  = 1;
    const int cpu_measure = 2;
#endif
    // const int gpu_warmup  = 5;
    const int gpu_measure = 1;

    float* h_A     = new float[M * N];
    float* h_B     = new float[N * K];
    float* h_C_CPU = new float[M * K];
    float* h_C_GPU = new float[M * K];

    cout << "Initializing matrices (" << M << " x " << N << " x " << K << ") ..." << endl;

    srand(42); // reproducible
    for (int i = 0; i < (int)(M * N); i++)
        h_A[i] = static_cast<float>(rand() % 100) / 10.0f;
    for (int i = 0; i < (int)(N * K); i++)
        h_B[i] = static_cast<float>(rand() % 100) / 10.0f;

#if CPU
    cout << "\n========== CPU Native (Scalar) Benchmark ==========\n";

    for (int i = 0; i < cpu_warmup; i++) {
        memset(h_C_CPU, 0, M * K * sizeof(float));
        matMulScalar(h_C_CPU, h_A, h_B);
    }

    double cpu_total_ms = 0.0;
    for (int iter = 0; iter < cpu_measure; iter++) {
        memset(h_C_CPU, 0, M * K * sizeof(float));
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

    cout << "\n========== Warp-Tiled Kernel Benchmark ==========\n";

    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    dim3 blockDim(NUM_THREADS);
    dim3 gridDim((K + BK - 1) / BK, (M + BM - 1) / BM);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // // Warm-up
    // cout << "Warm-up..." << endl;
    // for (int i = 0; i < gpu_warmup; i++) {
    //     CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    //     CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    //     CUDA_CHECK(cudaMemset(d_C, 0, sizeC));
    //     matMulWarpTiled<<<gridDim, blockDim>>>(d_C, d_A, d_B);
    //     CUDA_CHECK(cudaGetLastError());
    //     CUDA_CHECK(cudaDeviceSynchronize());
    // }

    // Measured runs
    float total_h2d = 0.0f, total_kernel = 0.0f, total_d2h = 0.0f;

    for (int iter = 0; iter < gpu_measure; iter++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float h2d_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));
        total_h2d += h2d_ms;

        CUDA_CHECK(cudaMemset(d_C, 0, sizeC));
        CUDA_CHECK(cudaEventRecord(start));
        matMulWarpTiled<<<gridDim, blockDim>>>(d_C, d_A, d_B);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
        total_kernel += kernel_ms;

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
    cout << "\nValidating GPU result against CPU..." << endl;
    validateCPU(h_C_CPU, h_C_GPU);
#endif

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