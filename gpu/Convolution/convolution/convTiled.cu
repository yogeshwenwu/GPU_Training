#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

#define CPU 1
#define TILE_SIZE 10
#define MASK_SIZE 7
#define SHARED_SIZE (TILE_SIZE + MASK_SIZE -1)
#define center ((MASK_SIZE - 1) / 2)

#define CUDA_CHECK(call)                                            \
    do                                                              \
    {                                                               \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess)                                     \
        {                                                           \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                 << " - " << cudaGetErrorString(err) << endl;       \
            return 1;                                               \
        }                                                           \
    } while (0)

// Define Constant Memory
__constant__ float mask_c[MASK_SIZE * MASK_SIZE];

void createMask(float *mask)
{
    for (int i = 0; i < (MASK_SIZE * MASK_SIZE); i++)
    {
        mask[i] = 1.0f / (MASK_SIZE * MASK_SIZE);
    }
}

// CPU Kernel (blur kernel)
void convolutionCPU(const uchar *input, float *mask, uchar *output, int width, int height)
{
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            float sum = 0.0f;
            for (int Mi = 0; Mi < MASK_SIZE; Mi++)
            {
                for (int Mj = 0; Mj < MASK_SIZE; Mj++)
                {
                    int ii = i + Mi - center;
                    int jj = j + Mj - center;

                    if (ii >= 0 && ii < (int)height && jj >= 0 && jj < (int)width)
                    {
                        sum += input[ii * width + jj] * mask[Mi * MASK_SIZE + Mj];
                    }
                }
            }
            output[i * width + j] = static_cast<uchar>(sum);//sum;
        }
    }
}

__global__ void convTiledGPU(const uchar *input, uchar *output, int width, int height)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // output index
    int outRow = blockIdx.y * TILE_SIZE + ty;
    int outCol = blockIdx.x * TILE_SIZE + tx;

    int inRow = outRow - center;
    int inCol = outCol - center;

    __shared__ uchar shInput[SHARED_SIZE * SHARED_SIZE];

    if(inRow >= 0 && inRow < height && inCol >=0 && inCol < width)
    {
        shInput[ty * SHARED_SIZE + tx] = input[inRow * width + inCol];
    }
    else
    {
        shInput[ty * SHARED_SIZE + tx] = 0;
    }
    __syncthreads();

    if (tx < TILE_SIZE && ty < TILE_SIZE)
    {
            float sum = 0.0f;
            for (int Mi = 0; Mi < MASK_SIZE; Mi++)
            {
                for (int Mj = 0; Mj < MASK_SIZE; Mj++)
                {
                        sum += shInput[(ty + Mi) * SHARED_SIZE + (tx + Mj)] * mask_c[Mi * MASK_SIZE + Mj];
                }
            }
            if(outRow < height && outCol < width)
            {
                output[outRow * width + outCol] = static_cast<uchar>(sum); //sum;
            }
    }
}

int main()
{
    // #if CPU
    //     const int cpu_warmup  = 1;
    //     const int cpu_measure = 2;
    // #endif
    // const int gpu_warmup  = 5;
    // const int gpu_measure = 50;

    // load image
    // cv::Mat img = cv::imread("./assets/img.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img = cv::imread("./assets/181279-flamingo_spreading_wings-greater_flamingo-american_flamingo-chilean_flamingo-ibis-3840x2160.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cerr << "Failed to load image: ./assets/img.jpg" << endl;
        return 1;
    }

    // align image
    if (!img.isContinuous())
    {
        img = img.clone();
    }

    // size of image
    const int width = img.cols;
    const int height = img.rows;
    const size_t imgBytes = width * height * sizeof(unsigned char); // or width * height * 1 (In Greyscale, 1 pixel = 1 byte)
    const size_t maskBytes = MASK_SIZE * MASK_SIZE * sizeof(float);

    // create Mask
    float mask[MASK_SIZE * MASK_SIZE];
    createMask(mask);

    #if CPU
        cv::Mat outImgCPU(height, width, CV_8UC1, cv::Scalar(0));
    #endif
    cv::Mat outImgGPU(height, width, CV_8UC1);

    // CPU code
    #if CPU
        // cout << "============ CPU Benchmark ============="<< endl;
        // // Warm-up run
        // for(int i = 0; i < cpu_warmup; i++)
        // {
        //     convolutionCPU(img.data, mask, outImgCPU.data, width, height);
        // }

        // double cpu_total_ms = 0.0;
        // // Actual run
        // for(int i = 0; i < cpu_measure; i++)
        // {
        //     auto start = high_resolution_clock::now();
            convolutionCPU(img.data, mask, outImgCPU.data, width, height);
    //         auto end = high_resolution_clock::now();

    //         duration<double, milli> elapsed = end - start;
    //         cpu_total_ms += elapsed.count();
    //         cout << "  Run " << i + 1 << ": " << fixed << setprecision(4) << elapsed.count() << " ms\n";
    //     }
    //     double cpu_avg_ms = cpu_total_ms / cpu_measure;

    //     cout << "---------------------------------------------------\n";
    //     cout << "CPU Average time     : " << fixed << setprecision(4) << cpu_avg_ms << " ms\n" << endl;
    #endif
    

    // save cpu result image
    if (!cv::imwrite("./assets/tile_convolution_cpu.jpg", outImgCPU))
    {
        cerr << "Failed to write: ./assets/tile_convolution_cpu.jpg" << endl;
        return 1;
    }
    cout << "Saved CPU result to ./assets/tile_convolution_cpu.jpg" << endl;

    // Device variable declaration
    uchar *d_input;
    uchar *d_output;
    // float *d_mask;

    CUDA_CHECK(cudaMalloc(&d_input, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_output, imgBytes));
    // CUDA_CHECK(cudaMalloc(&d_mask, maskBytes));

    // HTD code
    CUDA_CHECK(cudaMemcpy(d_input, img.data, imgBytes, cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_mask, mask, maskBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(mask_c, mask, maskBytes));
    CUDA_CHECK(cudaMemset(d_output, 0, imgBytes));

    // Launch Kernel
    dim3 blockDim(SHARED_SIZE, SHARED_SIZE);
    dim3 gridDim((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    convTiledGPU<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // DTH code
    CUDA_CHECK(cudaMemcpy(outImgGPU.data, d_output, imgBytes, cudaMemcpyDeviceToHost));

    // save gpu result image
    if (!cv::imwrite("./assets/tile_convolution_gpu.jpg", outImgGPU))
    {
        cerr << "Failed to write: ./assets/tile_convolution_gpu.jpg" << endl;
        return 1;
    }
    cout << "Saved GPU result to ./assets/tile_convolution_gpu.jpg" << endl;

    // Free memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}