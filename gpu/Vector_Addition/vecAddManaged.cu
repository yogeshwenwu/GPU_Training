#include <iostream>
#include <vector>

__global__ void vecAddKernel(int *vecRes, int *vecA, int *vecB){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    vecRes[i] = vecA[i] + vecB[i];
}

void vecAddNative(std::vector<int>& vecRes, int* vecA, int* vecB, int N){
    for(int i=0; i<N; i++)
    {
        vecRes[i] = vecA[i] + vecB[i];
    }
}

int main(){
    int N = 1000;
    std::vector<int> resG(N);

    int* devA = nullptr;
    int* devB = nullptr;
    int* devC = nullptr;

    cudaMallocManaged(&devA, N * sizeof(int));
    cudaMallocManaged(&devB, N * sizeof(int));
    cudaMallocManaged(&devC, N * sizeof(int));

    for(int i=0; i<N; i++){
        devA[i] = rand() % 100;
        devB[i] = rand() % 100;
    }

    // CPU run
    vecAddNative(resG, devA, devB, N);

    int threadsList[6] = {32, 64, 128, 256, 512, 1024};
    int threads = threadsList[3]; // 256
    int blocks = (N + threads -1)/threads; // 4

    // GPU run
    vecAddKernel<<<blocks, threads>>>(devC , devA, devB);
    cudaDeviceSynchronize();

    for(int i=0; i<N;i++){
        if(resG[i]!=devC[i])
        {
            std::cout << "Mismatch in index:" << i << "  native:" << resG[i] << " & GPU:" << devC[i];  
            return -1;
        }
    }
    std::cout << "Successfully verified!";

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
}