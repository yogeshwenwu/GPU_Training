#define CL_TARGET_OPENCL_VERSION 120

#include <iostream>
#include <cmath>
#include <CL/cl.h>
#include <chrono>
#include <iomanip>

#define DEBUG 0
#define WG_SIZE 16 // max work-group size can be 512 (32 * 32 =  1024) > 512, so (16 * 16 = 256) > 512
#define CPU 1

static const uint M = 1024;
static const uint N = 1024;
static const uint K = 1024;

// static const uint M = 4096;
// static const uint N = 4096;
// static const uint K = 4096;

using namespace std;
using namespace std::chrono;

// CPU Native Kernel
void MatMulScalar(float* matRes, float* matA, float* matB)
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

void checkBuildLog(cl_program program, cl_device_id device, cl_int err) {
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char *log = (char *)malloc(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        fprintf(stderr, "Kernel build failed:\n%s\n", log);
        free(log);
        exit(1);
    }
}

double eventTimeMs(cl_event event)
{
    cl_int err;
    cl_ulong start = 0;
    cl_ulong end = 0;

    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    if(err != CL_SUCCESS)
    {
        cout << "Failed to get event start time, error code: " << err << endl;
        exit(1);
    }

    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    if(err != CL_SUCCESS)
    {
        cout << "Failed to get event end time, error code: " << err << endl;
        exit(1);
    }

    return static_cast<double>(end - start) * 1e-6;
}

cl_ulong eventTimeNs(cl_event event, cl_profiling_info timeType)
{
    cl_int err;
    cl_ulong time = 0;

    err = clGetEventProfilingInfo(event, timeType, sizeof(time), &time, NULL);
    if(err != CL_SUCCESS)
    {
        cout << "Failed to get event profiling time, error code: " << err << endl;
        exit(1);
    }

    return time;
}

// GPU Native Kernel
const char* kernelSource = R"(
    __kernel void MatMulKernel(__global float* matRes, __global const float* matA, __global const float* matB, const uint M, const uint N, const uint K)
    {
        /*
        - Launch Matrix C Dimension (M * K) number of threads.
        - Each thread computes one element of the result matrix C, which is the dot product of one row of A and one column of B.
        */
        const uint WG_SIZE = 16;

        uint row = get_group_id(1) * WG_SIZE + (get_local_id(0) / WG_SIZE);
        uint col = get_group_id(0) * WG_SIZE + (get_local_id(0) % WG_SIZE);

        if (row < M && col < K)
        {
            float sum = 0.0f;
            for (int n = 0; n < N; n++)
            {
                sum += matA[row * N + n] * matB[n * K + col];
            }
            matRes[row * K + col] = sum;
        }
    }
)";

bool validateCPU(const float* cpu, const float* gpu, float tol = 1e-3f)
{
    int mismatches = 0;
    const int max_print = 10;   // print only first few mismatches

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
    // device variables
    cl_int err;
    char platformName[256];
    char deviceName[256];
    cl_platform_id platforms;
    cl_device_id devices;
    cl_context context;
    cl_command_queue queue;
    cl_program programs;
    cl_kernel kernel;
    cl_mem d_matA;
    cl_mem d_matB;
    cl_mem d_matRes;

    #if CPU
    const int cpu_warmup  = 3;
    const int cpu_measure = 2;
    #endif
    const int gpu_warmup = 5;
    const int gpu_measure = 50;

    // size of matrix
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeRes = M * K * sizeof(float);

    // Init matrices
    float* matrixA = new float[M * N];
    float* matrixB = new float[N * K];
    float* matrixResCPU = new float[M * K];
    float* matrixResGPU = new float[M * K];

    // initialize the input matrix
    cout << "Initializing matrices (" << M << "x" << N << "x" << K << ")..." << endl;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            matrixA[i * N + j] = static_cast<float>(N * i + j);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++)
            matrixB[i * K + j] = static_cast<float>(i + j);

    #if CPU
    cout << "\n========== CPU Native (Scalar) Benchmark ==========\n";

    for (int i = 0; i < cpu_warmup; i++)
    {
        MatMulScalar(matrixResCPU, matrixA, matrixB);
    }
    
    double cpu_total_ms = 0.0;

    for (int iter = 0; iter < cpu_measure; iter++)
    {
        auto start = high_resolution_clock::now();
        MatMulScalar(matrixResCPU, matrixA, matrixB);
        auto end = high_resolution_clock::now();

        duration<double, milli> elapsed = end - start;
        cpu_total_ms += elapsed.count();

        cout << "  Run " << iter + 1 << ": "
             << fixed << setprecision(2) << elapsed.count() << " ms" << endl;
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

    // DEVICE CODE
    cout << "\n========== Naive OpenCL Kernel Benchmark ==========\n";

    // find platform ID and Name
    err = clGetPlatformIDs(1, &platforms, NULL);
    if(err != CL_SUCCESS)
    {
        cout << "Failed to get platform" << endl;
        return 1;
    }

    err = clGetPlatformInfo(platforms, CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);
    if(err != CL_SUCCESS)
    {
        cout << "Failed to retrive platform name" << endl;
        return 1;
    }

    // find device ID and Name
    err = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, &devices, NULL);
    if(err != CL_SUCCESS)
    {
        cout << "Failed to get device" << endl;
        return 1;
    }

    err = clGetDeviceInfo(devices, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    if(err != CL_SUCCESS)
    {
        cout << "Failed to retrive device name" << endl;
        return 1;
    }

    // create context and queue
    context = clCreateContext(NULL, 1, &devices, NULL, NULL, &err);
    if(err != CL_SUCCESS)
    {
        cout << "Failed to create context" << endl;
        return 1;
    }

    queue = clCreateCommandQueue(context, devices, CL_QUEUE_PROFILING_ENABLE, &err);
    if(err != CL_SUCCESS)
    {
        cout << "Failed to create command queue" << endl;
        return 1;
    }

    // create opencl program and build program
    programs = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    if(err != CL_SUCCESS)
    {
        cout << "Failed to create program from source" << endl;
        return 1;
    }

    err = clBuildProgram(programs, 1, &devices, NULL, NULL, NULL);
    checkBuildLog(programs, devices, err);

    // initialize matrix
    d_matA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeA, NULL, &err);
    d_matB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeB, NULL, &err);
    d_matRes = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeRes, NULL, &err);    

    // create kernel from program
    kernel = clCreateKernel(programs, "MatMulKernel", &err);
    if(err != CL_SUCCESS)
    {
        cout << "Failed to create kernel from program" << endl;
        return 1;
    }

    // Set kernel args
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_matRes);
    if(err != CL_SUCCESS)
    {
        cout << "Failed to set arguments of kernel" << endl;
        return 1;
    }
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_matA);
    if(err != CL_SUCCESS)
    {
        cout << "Failed to set arguments of kernel" << endl;
        return 1;
    }
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_matB);
    if(err != CL_SUCCESS)
    {
        cout << "Failed to set arguments of kernel" << endl;
        return 1;
    }
    err = clSetKernelArg(kernel, 3, sizeof(uint), &M);
    if(err != CL_SUCCESS)
    {
        cout << "Failed to set arguments of kernel" << endl;
        return 1;
    }
    err = clSetKernelArg(kernel, 4, sizeof(uint), &N);
    if(err != CL_SUCCESS)
    {
        cout << "Failed to set arguments of kernel" << endl;
        return 1;
    }
    err = clSetKernelArg(kernel, 5, sizeof(uint), &K);
    if(err != CL_SUCCESS)
    {
        cout << "Failed to set arguments of kernel" << endl;
        return 1;
    }

    // Kernel launch configuration
    const size_t grid_x = (K + WG_SIZE - 1) / WG_SIZE;
    const size_t grid_y = (M + WG_SIZE - 1) / WG_SIZE;

    const size_t local_size[2] = {
        WG_SIZE * WG_SIZE,
        1
    };

    const size_t global_size[2] = {
        grid_x * local_size[0],
        grid_y * local_size[1]
    };


    // Warmup run
    cout << "Warm-up..." << endl;
    for (int i = 0; i < gpu_warmup; i++)
    {
        err = clEnqueueWriteBuffer(queue, d_matA, CL_TRUE, 0, sizeA, matrixA, 0, NULL, NULL);
        if(err != CL_SUCCESS)
        {
            cout << "Failed to copy matrix A from Host to Device during warmup" << endl;
            return 1;
        }

        err = clEnqueueWriteBuffer(queue, d_matB, CL_TRUE, 0, sizeB, matrixB, 0, NULL, NULL);
        if(err != CL_SUCCESS)
        {
            cout << "Failed to copy matrix B from Host to Device during warmup" << endl;
            return 1;
        }

        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
        if(err != CL_SUCCESS)
        {
            cout << "Failed to launch warmup kernel" << err << endl;
            return 1;
        }

        err = clFinish(queue);
        if(err != CL_SUCCESS)
        {
            cout << "Failed to sync devices during warmup" << endl;
            return 1;
        }

        err = clEnqueueReadBuffer(queue, d_matRes, CL_TRUE, 0, sizeRes, matrixResGPU, 0, NULL, NULL);
        if(err != CL_SUCCESS)
        {
            cout << "Failed to copy results from Device to Host during warmup" << endl;
            return 1;
        }
    }

    // Actual run
    double total_h2d = 0.0;
    double total_kernel = 0.0;
    double total_d2h = 0.0;

    for (int iter = 0; iter < gpu_measure; iter++)
    {
        cl_event writeEventA;
        cl_event writeEventB;
        cl_event kernelEvent;
        cl_event readEvent;

        // H2D 
        err = clEnqueueWriteBuffer(queue, d_matA, CL_FALSE, 0, sizeA, matrixA, 0, NULL, &writeEventA);
        if(err != CL_SUCCESS)
        {
            cout << "Failed to copy matrix A from Host to Device" << endl;
            return 1;
        }
        err = clEnqueueWriteBuffer(queue, d_matB, CL_FALSE, 0, sizeB, matrixB, 0, NULL, &writeEventB);
        if(err != CL_SUCCESS)
        {
            cout << "Failed to copy matrix B from Host to Device" << endl;
            return 1;
        }

        cl_event writeEvents[2] = {writeEventA, writeEventB};
        err = clWaitForEvents(2, writeEvents);
        if(err != CL_SUCCESS)
        {
            cout << "Failed to wait for H2D events" << endl;
            return 1;
        }

        cl_ulong h2d_start = eventTimeNs(writeEventA, CL_PROFILING_COMMAND_START);
        cl_ulong h2d_end = eventTimeNs(writeEventB, CL_PROFILING_COMMAND_END);
        double h2d_ms = static_cast<double>(h2d_end - h2d_start) * 1e-6;
        total_h2d += h2d_ms;

        // Launch Kernel and synchronize
        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, &kernelEvent);
        if(err != CL_SUCCESS)
        {
            cout << "Failed to launch the kernel" << err << endl;
            return 1;
        }

        err = clWaitForEvents(1, &kernelEvent);
        if(err != CL_SUCCESS)
        {
            cout << "Failed to sync devices" << endl;
            return 1;
        }

        double kernel_ms = eventTimeMs(kernelEvent);
        total_kernel += kernel_ms;

        // D2H
        err = clEnqueueReadBuffer(queue, d_matRes, CL_FALSE, 0, sizeRes, matrixResGPU, 0, NULL, &readEvent);
        if(err != CL_SUCCESS)
        {
            cout << "Failed to copy results from Device to Host" << endl;
            return 1;
        }

        err = clWaitForEvents(1, &readEvent);
        if(err != CL_SUCCESS)
        {
            cout << "Failed to wait for D2H event" << endl;
            return 1;
        }

        double d2h_ms = eventTimeMs(readEvent);
        total_d2h += d2h_ms;

        cout << "  Run " << iter + 1 
             << " | H2D: " << fixed << setprecision(3) << h2d_ms
             << " ms | Kernel: " << kernel_ms
             << " ms | D2H: " << d2h_ms << " ms" << endl;

        clReleaseEvent(writeEventA);
        clReleaseEvent(writeEventB);
        clReleaseEvent(kernelEvent);
        clReleaseEvent(readEvent);
    }

    double avg_h2d    = total_h2d    / gpu_measure;
    double avg_kernel = total_kernel / gpu_measure;
    double avg_d2h    = total_d2h    / gpu_measure;
    double avg_total  = avg_h2d + avg_kernel + avg_d2h;

    double gpu_gflops = (flops / (avg_kernel * 1e-3)) / 1e9;

    #if CPU
    // Validate 
    cout << "\nValidating GPU result against CPU..." << endl;
    validateCPU(matrixResCPU, matrixResGPU);
    #endif

    // Benchmark Results
    cout << "\n---------------------------------------------------\n";
    cout << "Average H2D time     : " << fixed << setprecision(3) << avg_h2d    << " ms\n";
    cout << "Average Kernel time  : " << fixed << setprecision(3) << avg_kernel << " ms\n";
    cout << "Average D2H time     : " << fixed << setprecision(3) << avg_d2h    << " ms\n";
    cout << "Average Total time   : " << fixed << setprecision(3) << avg_total  << " ms\n";
    cout << "---------------------------------------------------\n";
    cout << "Pure Kernel GFLOPS   : " << fixed << setprecision(2) << gpu_gflops << " GFLOPS\n";
    #if CPU
    cout << "Speedup (Kernel vs CPU): " << fixed << setprecision(2)
         << (cpu_avg_ms / avg_kernel) << "x\n";
    #endif
    cout << "===================================================\n";

    // Cleanup
    clReleaseMemObject(d_matA);
    clReleaseMemObject(d_matB);
    clReleaseMemObject(d_matRes);
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(programs);
    clReleaseKernel(kernel);

    delete[] matrixA;
    delete[] matrixB;
    delete[] matrixResCPU;
    delete[] matrixResGPU;

    cout << "\nBenchmark complete." << endl;
    return 0;
}
