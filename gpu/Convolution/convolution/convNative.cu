#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace std;

#define MASK_SIZE 7
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

__global__ void convNativeGPU(const uchar *input, const float* mask, uchar *output, int width, int height)
{
    int globalRow = blockDim.y * blockIdx.y + threadIdx.y;
    int globalCol = blockDim.x * blockIdx.x + threadIdx.x;

    if (globalRow < height && globalCol < width)
    {
        float sum = 0.0f;
        for (int Mi = 0; Mi < MASK_SIZE; Mi++)
        {
            for (int Mj = 0; Mj < MASK_SIZE; Mj++)
            {
                int row = globalRow + Mi - center;
                int col = globalCol + Mj - center;

                if (row >= 0 && row < height && col >= 0 && col < width)
                {
                    sum += input[row * width + col] * mask[Mi * MASK_SIZE + Mj];
                }
            }
        }
        output[globalRow * width + globalCol] = static_cast<uchar>(sum); //sum;
    }
}


int main()
{
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

    cv::Mat outImgCPU(height, width, CV_8UC1);
    cv::Mat outImgGPU(height, width, CV_8UC1);

    // CPU run
    convolutionCPU(img.data, mask, outImgCPU.data, width, height);

    // save cpu result image
    if (!cv::imwrite("./assets/convolution_cpu.jpg", outImgCPU))
    {
        cerr << "Failed to write: ./assets/convolution_cpu.jpg" << endl;
        return 1;
    }
    cout << "Saved CPU result to ./assets/convolution_cpu.jpg" << endl;

    // Device variable declaration
    uchar *d_input;
    uchar *d_output;
    float *d_mask;

    CUDA_CHECK(cudaMalloc(&d_input, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_output, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_mask, maskBytes));

    // HTD code
    CUDA_CHECK(cudaMemcpy(d_input, img.data, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, mask, maskBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, imgBytes));

    // Launch Kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    convNativeGPU<<<gridDim, blockDim>>>(d_input, d_mask, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // DTH code
    CUDA_CHECK(cudaMemcpy(outImgGPU.data, d_output, imgBytes, cudaMemcpyDeviceToHost));

    // save gpu result image
    if (!cv::imwrite("./assets/convolution_gpu.jpg", outImgGPU))
    {
        cerr << "Failed to write: ./assets/convolution_gpu.jpg" << endl;
        return 1;
    }
    cout << "Saved GPU result to ./assets/convolution_gpu.jpg" << endl;

    // Free memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_mask));
}