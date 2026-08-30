#include <iostream>
#include <vector>
#include <cmath>
#include <cuda/cmath>
#include <cuda_runtime.h>
#include <chrono>

// // Integer implementation
// __global__ void vecAddKernel(int *vecRes, int *vecA, int *vecB){
//     int i = blockDim.x * blockIdx.x + threadIdx.x;

//     vecRes[i] = vecA[i] + vecB[i];
// }

__global__ void vecAddKernel(float *vecRes, float *vecA, float *vecB){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    vecRes[i] = vecA[i] + vecB[i];
}

// void vecAddNative(std::vector<int>& vecRes, std::vector<int>& vecA, std::vector<int>& vecB, int N){
//     for(int i=0; i<N; i++)
//     {
//         vecRes[i] = vecA[i] + vecB[i];
//     }
// }

void vecAddNative(std::vector<float>& vecRes, std::vector<float>& vecA, std::vector<float>& vecB, int N){
    for(int i=0; i<N; i++)
    {
        vecRes[i] = vecA[i] + vecB[i];
    }
}

int main(){
    // init host variables
    int N = 1<<26;
    int warmup_iter = 1;

    // std::vector<int> arr(N);
    // std::vector<int> brr(N);
    // std::vector<int> resC(N);
    // std::vector<int> resG(N);

    std::vector<float> arr(N);
    std::vector<float> brr(N);
    std::vector<float> resC(N);
    std::vector<float> resG(N);
    
    // initialize vector
    for(int i=0; i<N; i++){
        arr[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        brr[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    }
    
    // It's CPU Time!!
    // warmup run
    // for(int i=0; i<warmup_iter; i++)
    // {
    //     vecAddNative(resC, arr, brr, N);
    // }
    // memset(resC.data(), 0, resC.size() * sizeof(float));

    // Actual run
    auto start = std::chrono::high_resolution_clock::now();
    vecAddNative(resC, arr, brr, N);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // GPU Time!!
    // int* devA = nullptr;
    // int* devB = nullptr;
    // int* devC = nullptr;
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    cudaMalloc(&devA, N * sizeof(float));
    cudaMalloc(&devB, N * sizeof(float));
    cudaMalloc(&devC, N * sizeof(float));

    cudaMemcpy(devA, arr.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, brr.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(devC, 0, N*sizeof(float));

    int threadsList[6] = {32, 64, 128, 256, 512, 1024};
    int threads = threadsList[3]; // 256
    int blocks = (N + threads -1)/threads; // 4

    // warmup run
    for(int i=0; i<warmup_iter; i++)
    {
        vecAddKernel<<<blocks, threads>>>(devC , devA, devB);
    }
    cudaMemset(devC, 0, N * sizeof(float));
    
    // Actual run
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
    vecAddKernel<<<blocks, threads>>>(devC , devA, devB);
    cudaDeviceSynchronize();
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, startEvent, stopEvent);
    cudaMemcpy(resG.data(), devC, N*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i<N;i++){
        if(std::abs(resG[i] - resC[i]) > 1e-5)
        {
            std::cout << "Mismatch in index:" << i << "  native:" << resC[i] << " & GPU:" << resG[i];  
            return -1;
        }
        // std::cout << "i= " << i << "; A[i]: " << arr[i] << "; B[i]: "<< brr[i] << "; CPU Result: " << resC[i] << "; GPU Result: " << resG[i] << std::endl;
    }
    std::cout << "Successfully verified!\n";
    std::cout << "CPU Time: " << cpu_time.count() << " milliseconds\n";
    std::cout << "GPU Time: " << gpu_time << " milliseconds\n";

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
}