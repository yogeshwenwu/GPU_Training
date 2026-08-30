#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

#define MASK_SIZE 15

void printDiffStats(const string& label, const cv::Mat& a, const cv::Mat& b, const string& outPath)
{
    if (a.size() != b.size())
    {
        cerr << "Size mismatch (" << label << "): " << a.size() << " vs " << b.size() << endl;
        return;
    }

    cv::Mat diff;
    cv::absdiff(a, b, diff);

    double minVal, maxVal;
    cv::minMaxLoc(diff, &minVal, &maxVal);
    cv::Scalar meanVal = cv::mean(diff);

    cout << "\n--- " << label << " ---" << endl;
    cout << "Max diff:  " << maxVal << endl;
    cout << "Mean diff: " << meanVal[0] << endl;

    cv::imwrite(outPath, diff);
}

int main()
{
    // Load all images as grayscale
    cv::Mat input = cv::imread("./assets/img.jpg", cv::IMREAD_GRAYSCALE);
    // cv::Mat cpu_res = cv::imread("./assets/convolution_cpu.jpg", cv::IMREAD_GRAYSCALE);
    // cv::Mat gpu_res = cv::imread("./assets/convolution_gpu.jpg", cv::IMREAD_GRAYSCALE);

    // cv::Mat cpu_res = cv::imread("./assets/const_convolution_cpu.jpg", cv::IMREAD_GRAYSCALE);
    // cv::Mat gpu_res = cv::imread("./assets/const_convolution_gpu.jpg", cv::IMREAD_GRAYSCALE);

    cv::Mat cpu_res = cv::imread("./assets/tile_convolution_cpu.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat gpu_res = cv::imread("./assets/tile_convolution_gpu.jpg", cv::IMREAD_GRAYSCALE);

    if (input.empty() || cpu_res.empty() || gpu_res.empty())
    {
        cerr << "Failed to load one of the images" << endl;
        return 1;
    }

    // Save grayscale version of original input
    if (!cv::imwrite("./assets/img_gray.jpg", input))
    {
        cerr << "Failed to write: ./assets/img_gray.jpg" << endl;
        return 1;
    }
    cout << "Saved grayscale input to ./assets/img_gray.jpg" << endl;

    // --- OpenCV reference box blur ---
    // BORDER_CONSTANT with value 0 matches your zero-padding CPU/GPU border policy.
    // normalize=true divides by MASK_SIZE*MASK_SIZE, same as your 1/(MASK_SIZE*MASK_SIZE) mask.
    cv::Mat opencv_res;
    cv::boxFilter(
        input,
        opencv_res,
        -1,                                  // same depth as input (CV_8U)
        cv::Size(MASK_SIZE, MASK_SIZE),
        cv::Point(-1, -1),                   // anchor at kernel center
        true,                                 // normalize
        cv::BORDER_CONSTANT                   // zero padding at borders
    );

    if (!cv::imwrite("./assets/opencv_convolution.jpg", opencv_res))
    {
        cerr << "Failed to write: ./assets/opencv_convolution.jpg" << endl;
        return 1;
    }
    cout << "Saved OpenCV blur result to ./assets/opencv_convolution.jpg" << endl;

    // Check sizes against input
    if (input.size() != cpu_res.size())
    {
        cerr << "Size mismatch with CPU: " << input.size() << " vs " << cpu_res.size() << endl;
        return 1;
    }
    if (input.size() != gpu_res.size())
    {
        cerr << "Size mismatch with GPU: " << input.size() << " vs " << gpu_res.size() << endl;
        return 1;
    }
    if (input.size() != opencv_res.size())
    {
        cerr << "Size mismatch with OpenCV: " << input.size() << " vs " << opencv_res.size() << endl;
        return 1;
    }

    // Pairwise comparisons
    printDiffStats("Input vs CPU",   input,   cpu_res,    "./assets/diff_cpu.jpg");
    printDiffStats("Input vs GPU",   input,   gpu_res,    "./assets/diff_gpu.jpg");
    printDiffStats("Input vs OpenCV", input,  opencv_res, "./assets/diff_opencv.jpg");

    printDiffStats("CPU vs GPU",     cpu_res, gpu_res,    "./assets/diff_cpu_gpu.jpg");
    printDiffStats("CPU vs OpenCV",  cpu_res, opencv_res, "./assets/diff_cpu_opencv.jpg");
    printDiffStats("GPU vs OpenCV",  gpu_res, opencv_res, "./assets/diff_gpu_opencv.jpg");

    cout << "\nSaved diff images to ./assets/" << endl;

    return 0;
}

// #include <opencv2/opencv.hpp>
// #include <iostream>

// using namespace std;

// int main()
// {
//     // Load all images as grayscale
//     cv::Mat input = cv::imread(
//         "./assets/img.jpg",
//         cv::IMREAD_GRAYSCALE
//     );

//     cv::Mat cpu_res = cv::imread(
//         "./assets/const_convolution_cpu.jpg",
//         cv::IMREAD_GRAYSCALE
//     );

//     cv::Mat gpu_res = cv::imread(
//         "./assets/const_convolution_gpu.jpg",
//         cv::IMREAD_GRAYSCALE
//     );

//     if (input.empty() || cpu_res.empty() || gpu_res.empty())
//     {
//         cerr << "Failed to load one of the images" << endl;
//         return 1;
//     }

//     // Save grayscale version of original input
//     if (!cv::imwrite("./assets/img_gray.jpg", input))
//     {
//         cerr << "Failed to write: ./assets/img_gray.jpg" << endl;
//         return 1;
//     }

//     cout << "Saved grayscale input to ./assets/img_gray.jpg" << endl;

//     // Check sizes
//     if (input.size() != cpu_res.size())
//     {
//         cerr << "Size mismatch with CPU: "
//              << input.size()
//              << " vs "
//              << cpu_res.size()
//              << endl;

//         return 1;
//     }

//     if (input.size() != gpu_res.size())
//     {
//         cerr << "Size mismatch with GPU: "
//              << input.size()
//              << " vs "
//              << gpu_res.size()
//              << endl;

//         return 1;
//     }

//     // Input vs CPU
//     cv::Mat cpu_diff;
//     cv::absdiff(input, cpu_res, cpu_diff);

//     double minValCPU, maxValCPU;
//     cv::minMaxLoc(cpu_diff, &minValCPU, &maxValCPU);

//     cv::Scalar meanCPU = cv::mean(cpu_diff);

//     cout << "\n--- Input vs CPU ---" << endl;
//     cout << "Max diff:  " << maxValCPU << endl;
//     cout << "Mean diff: " << meanCPU[0] << endl;

//     cv::imwrite(
//         "./assets/diff_cpu.jpg",
//         cpu_diff
//     );

//     // Input vs GPU
//     cv::Mat gpu_diff;
//     cv::absdiff(input, gpu_res, gpu_diff);

//     double minValGPU, maxValGPU;
//     cv::minMaxLoc(gpu_diff, &minValGPU, &maxValGPU);

//     cv::Scalar meanGPU = cv::mean(gpu_diff);

//     cout << "\n--- Input vs GPU ---" << endl;
//     cout << "Max diff:  " << maxValGPU << endl;
//     cout << "Mean diff: " << meanGPU[0] << endl;

//     cv::imwrite(
//         "./assets/diff_gpu.jpg",
//         gpu_diff
//     );

//     // CPU vs GPU
//     cv::Mat cpu_gpu_diff;
//     cv::absdiff(cpu_res, gpu_res, cpu_gpu_diff);

//     double minValCPUGPU, maxValCPUGPU;
//     cv::minMaxLoc(
//         cpu_gpu_diff,
//         &minValCPUGPU,
//         &maxValCPUGPU
//     );

//     cv::Scalar meanCPUGPU = cv::mean(cpu_gpu_diff);

//     cout << "\n--- CPU vs GPU ---" << endl;
//     cout << "Max diff:  " << maxValCPUGPU << endl;
//     cout << "Mean diff: " << meanCPUGPU[0] << endl;

//     cv::imwrite(
//         "./assets/diff_cpu_gpu.jpg",
//         cpu_gpu_diff
//     );

//     cout << "\nSaved:" << endl;
//     cout << "./assets/diff_cpu.jpg" << endl;
//     cout << "./assets/diff_gpu.jpg" << endl;
//     cout << "./assets/diff_cpu_gpu.jpg" << endl;

//     return 0;
// }