#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#define CPU 1
#define N (1 << 18)
#define LOCAL_SIZE 256

using namespace std;
using namespace std::chrono;

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

double eventElapsedMs(cl_event event)
{
    cl_ulong start = 0;
    cl_ulong end = 0;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    return static_cast<double>(end - start) * 1.0e-6;
}

double eventRangeMs(cl_event first, cl_event last)
{
    cl_ulong start = 0;
    cl_ulong end = 0;
    clGetEventProfilingInfo(first, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(last, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    return static_cast<double>(end - start) * 1.0e-6;
}

// we can't directly define kernel as CUDA, as we are compiling with g++ not nvcc. 
// __kernel is not a known keyword in g++. 
const char* kernelSource = R"(
    __kernel void vecAdd(__global const float* vecA, __global const float* vecB, __global float* vecRes, const int N)
    {
        int i = get_global_id(0);
        if(i < N)
        {
            vecRes[i] = vecA[i] + vecB[i];   
        }   
    }
)";

// CPU function
void vecAddCPU(float* A, float* B, float* C)
{
    for(int i=0; i<N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

// verify 
bool verify(float* C, float* G)
{
    for(int i=0; i<N; i++)
    {
        if(C[i]!= G[i])
        {
            cout << "Mismatch at index: " << i << " CPU: " << C[i] << " vs GPU: " << G[i] << endl; 
            return false;
        }
    }
    return true;
}

int main()
{
#if CPU
    const int cpu_warmup = 1;
    const int cpu_measure = 3;
#endif
    const int gpu_warmup = 5;
    const int gpu_measure = 50;

    cl_int err;
    char deviceName[256];
    char platformName[256];

    cout << "Initializing vectors (N = " << N << ")..." << endl;

    // define host variable 
    const int n = N;
    float h_vecA[N];
    float h_vecB[N];
    float h_vecResCPU[N];
    float h_vecResGPU[N];
    
    // initialize input vectors
    for(int i = 0; i < N; i++)
    {
        h_vecA[i] = rand() % 100;
        h_vecB[i] = rand() % 100;
    }

#if CPU
    cout << "\n========== CPU Native Benchmark ==========\n";

    for(int i = 0; i < cpu_warmup; i++)
    {
        vecAddCPU(h_vecA, h_vecB, h_vecResCPU);
    }

    double cpu_total_ms = 0.0;
    for(int iter = 0; iter < cpu_measure; iter++)
    {
        auto start = high_resolution_clock::now();
        vecAddCPU(h_vecA, h_vecB, h_vecResCPU);
        auto end = high_resolution_clock::now();

        duration<double, milli> elapsed = end - start;
        cpu_total_ms += elapsed.count();

        cout << "  Run " << iter + 1 << ": "
             << fixed << setprecision(3) << elapsed.count() << " ms" << endl;
    }

    double cpu_avg_ms = cpu_total_ms / cpu_measure;
#else
    vecAddCPU(h_vecA, h_vecB, h_vecResCPU);
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

    cout << "\n========== OpenCL Vector Add Benchmark ==========\n";

    // Identify platform and device
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if(err != CL_SUCCESS) {
        cout << "Failed to get platform, error code: " << err << endl;
        return 1;
    }

    // Print platform name
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);
    if(err != CL_SUCCESS) {
        cout << "Failed to get platform name, error code: " << err << endl;
        return 1;
    }
    cout << "Platform: " << platformName << endl;

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if(err != CL_SUCCESS) {
        cout << "Failed to get device, error code: " << err << endl;
        return 1;
    }

    // print device name
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    if(err != CL_SUCCESS) {
        cout << "Failed to get device name, error code: " << err << endl;
        return 1;
    }
    cout << "Device: " << deviceName << endl;

    // create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if(err != CL_SUCCESS) {
        cout << "Failed to create context, error code: " << err << endl;
        return 1;
    }

    // create queue
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if(err != CL_SUCCESS) {
        cout << "Failed to create queue, error code: " << err << endl;
        return 1;
    }

    // define Device variable
    cl_mem d_vecA = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &err);
    if(err != CL_SUCCESS) {
        cout << "Failed to create input buffer for vecA, error code: " << err << endl;
        return 1;
    }

    cl_mem d_vecB = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &err);
    if(err != CL_SUCCESS) {
        cout << "Failed to create input buffer for vecB, error code: " << err << endl;
        return 1;
    }

    cl_mem d_vecRes = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, &err);
    if(err != CL_SUCCESS) {
        cout << "Failed to create input buffer for vecRes, error code: " << err << endl;
        return 1;
    }

    // Compile kernel at runtime
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    if(err != CL_SUCCESS) {
        cout << "Failed to create program from source, error code: " << err << endl;
        return 1;
    }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    checkBuildLog(program, device, err);

    // assign kernel to a var
    cl_kernel kernel = clCreateKernel(program, "vecAdd", &err);
    if(err != CL_SUCCESS) {
        cout << "Failed to create kernel, error code: " << err << endl;
        return 1;
    }

    // set kernel args
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_vecA);
    if(err != CL_SUCCESS) {
        cout << "Failed to set kernel arg 0, error code: " << err << endl;
        return 1;
    }
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_vecB);
    if(err != CL_SUCCESS) {
        cout << "Failed to set kernel arg 1, error code: " << err << endl;
        return 1;
    }
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_vecRes);
    if(err != CL_SUCCESS) {
        cout << "Failed to set kernel arg 2, error code: " << err << endl;
        return 1;
    }
    err = clSetKernelArg(kernel, 3, sizeof(int), &n);
    if(err != CL_SUCCESS) {
        cout << "Failed to set kernel arg 3, error code: " << err << endl;
        return 1;
    }

    // launch kernel
    size_t global_size = N;
    size_t local_size = LOCAL_SIZE;

    for(int i = 0; i < gpu_warmup; i++)
    {
        err = clEnqueueWriteBuffer(queue, d_vecA, CL_TRUE, 0, N * sizeof(float), h_vecA, 0, NULL, NULL);
        if(err != CL_SUCCESS) {
            cout << "Failed to copy input vecA from HTD during warmup, error code: " << err << endl;
            return 1;
        }

        err = clEnqueueWriteBuffer(queue, d_vecB, CL_TRUE, 0, N * sizeof(float), h_vecB, 0, NULL, NULL);
        if(err != CL_SUCCESS) {
            cout << "Failed to copy input vecB from HTD during warmup, error code: " << err << endl;
            return 1;
        }

        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
        if(err != CL_SUCCESS) {
            cout << "Failed to launch kernel during warmup, error code: " << err << endl;
            return 1;
        }

        err = clEnqueueReadBuffer(queue, d_vecRes, CL_TRUE, 0, N * sizeof(float), h_vecResGPU, 0, NULL, NULL);
        if(err != CL_SUCCESS) {
            cout << "Failed to copy result from DTH during warmup, error code: " << err << endl;
            return 1;
        }
    }
    clFinish(queue);

    double total_h2d = 0.0;
    double total_kernel = 0.0;
    double total_d2h = 0.0;

    for(int iter = 0; iter < gpu_measure; iter++)
    {
        cl_event writeAEvent = NULL;
        cl_event writeBEvent = NULL;
        cl_event kernelEvent = NULL;
        cl_event readEvent = NULL;

        // HTD copy
        err = clEnqueueWriteBuffer(queue, d_vecA, CL_FALSE, 0, N * sizeof(float), h_vecA, 0, NULL, &writeAEvent);
        if(err != CL_SUCCESS) {
            cout << "Failed to copy input vecA from HTD, error code: " << err << endl;
            return 1;
        }
        
        err = clEnqueueWriteBuffer(queue, d_vecB, CL_FALSE, 0, N * sizeof(float), h_vecB, 0, NULL, &writeBEvent);
        if(err != CL_SUCCESS) {
            cout << "Failed to copy input vecB from HTD, error code: " << err << endl;
            return 1;
        }

        err = clWaitForEvents(1, &writeBEvent);
        if(err != CL_SUCCESS) {
            cout << "Failed while waiting for HTD copy, error code: " << err << endl;
            return 1;
        }
        double h2d_ms = eventRangeMs(writeAEvent, writeBEvent);
        total_h2d += h2d_ms;

        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &kernelEvent);
        if(err != CL_SUCCESS) {
            cout << "Failed to launch kernel, error code: " << err << endl;
            return 1;
        }

        err = clWaitForEvents(1, &kernelEvent);
        if(err != CL_SUCCESS) {
            cout << "Failed while waiting for kernel, error code: " << err << endl;
            return 1;
        }
        double kernel_ms = eventElapsedMs(kernelEvent);
        total_kernel += kernel_ms;
        
        // DTH copy
        err = clEnqueueReadBuffer(queue, d_vecRes, CL_FALSE, 0, N * sizeof(float), h_vecResGPU, 0, NULL, &readEvent);
        if(err != CL_SUCCESS) {
            cout << "Failed to copy result from DTH, error code: " << err << endl;
            return 1;
        }

        err = clWaitForEvents(1, &readEvent);
        if(err != CL_SUCCESS) {
            cout << "Failed while waiting for DTH copy, error code: " << err << endl;
            return 1;
        }
        double d2h_ms = eventElapsedMs(readEvent);
        total_d2h += d2h_ms;

        cout << "  Run " << iter + 1
             << " | H2D: " << fixed << setprecision(3) << h2d_ms
             << " ms | Kernel: " << kernel_ms
             << " ms | D2H: " << d2h_ms << " ms" << endl;

        clReleaseEvent(writeAEvent);
        clReleaseEvent(writeBEvent);
        clReleaseEvent(kernelEvent);
        clReleaseEvent(readEvent);
    }

    double avg_h2d = total_h2d / gpu_measure;
    double avg_kernel = total_kernel / gpu_measure;
    double avg_d2h = total_d2h / gpu_measure;
    double avg_total = avg_h2d + avg_kernel + avg_d2h;

    double gpu_gflops = (ops / (avg_kernel * 1e-3)) / 1e9;
    double gpu_bandwidth = (bytes / (avg_kernel * 1e-3)) / 1e9;

    if(!verify(h_vecResCPU, h_vecResGPU))
    {
        return 1;
    }
    cout << "All test passed" << endl;

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

    cout << "\nBenchmark complete." << endl;

    // cleanup
    clReleaseMemObject(d_vecA);
    clReleaseMemObject(d_vecB);
    clReleaseMemObject(d_vecRes);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
