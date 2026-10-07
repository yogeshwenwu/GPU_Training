# GPU_Training

A hands-on repository for learning and experimenting with parallel programming across CPU, CUDA GPU, and OpenCL workloads. The project focuses on understanding performance, optimization strategies, memory access patterns, and kernel design through real code examples.

This repository contains:
- CPU examples using scalar, loop-unrolled, SSE, and AVX implementations
- CUDA GPU examples for vector addition and matrix multiplication
- OpenCL examples for cross-platform parallel computation
- Benchmarking code to compare execution time and correctness

Note: The OpenCL examples are present in the `dev` branch.

---

## Repository Structure

```text
GPU_Training/
├── cpu/
│   ├── MatrixMultiplication/
│   │   ├── MatMulAVX.cpp
│   │   ├── MatrixMultiplication.cpp
│   │   └── ...
│   ├── MatrixTranspose/
│   │   ├── MatrixTranspose.cpp
│   │   ├── MatrixTransposePadding.cpp
│   │   └── ...
│   ├── VectorAddition/
│   │   └── vectorAddition.cpp
│   └── compatability.cpp
├── gpu/
│   ├── Convolution/
│   │   └── convolution/
│   ├── Matrix Multiplication/
│   │   ├── matMul2DTileVector.cu
│   │   ├── matMul2DTiled.cu
│   │   ├── matMulCoalesced.cu
│   │   ├── matMulNative.cu
│   │   ├── matMulShared.cu
│   │   ├── matMulTile.cu
│   │   └── matmulWarp.cu
│   ├── Parallel_Reduction/
│   │   └── reduction/
│   └── Vector_Addition/
│       ├── vecAddManaged.cu
│       └── vectorAddition.cu
├── OpenCL/
│   ├── MatMul/
│   └── vectorAddition/
├── README.md
└── ...
```

---

## What is covered?

### CPU Examples
These examples focus on:
- scalar implementation
- loop unrolling
- SIMD intrinsics (SSE/AVX)
- matrix multiplication and transpose performance

Files include:
- `cpu/VectorAddition/vectorAddition.cpp`
- `cpu/MatrixMultiplication/MatrixMultiplication.cpp`
- `cpu/MatrixMultiplication/MatMulAVX.cpp`
- `cpu/MatrixTranspose/MatrixTranspose.cpp`
- `cpu/MatrixTranspose/MatrixTransposePadding.cpp`

### CUDA GPU Examples
These examples focus on:
- parallel kernel launch
- vector addition on GPU
- matrix multiplication kernels
- tiled and shared-memory optimization
- benchmarking and validation

Files include:
- `gpu/Vector_Addition/vectorAddition.cu`
- `gpu/Matrix Multiplication/matMulNative.cu`
- `gpu/Matrix Multiplication/matMulShared.cu`
- `gpu/Matrix Multiplication/matMulTile.cu`
- `gpu/Matrix Multiplication/matmulWarp.cu`

### OpenCL Examples
The repo contains OpenCL code under `OpenCL/`, including:
- `OpenCL/vectorAddition`
- `OpenCL/MatMul`

These are useful for understanding a portable heterogeneous computing model similar to CUDA but not tied to NVIDIA hardware.

---

## Prerequisites

### For CPU programs
- GCC / G++ installed
- A system with AVX/SSE support if using vectorized code

Check installation:
```bash
g++ --version
```

### For CUDA programs
- NVIDIA CUDA Toolkit installed
- `nvcc` available in PATH
- NVIDIA GPU with CUDA support

Check installation:
```bash
nvcc --version
nvidia-smi
```

### For OpenCL programs
- OpenCL headers and libraries installed
- An OpenCL-capable GPU or CPU runtime
- A compiler such as `gcc` or `clang`

Check installation:
```bash
clinfo
```

---

## Clone the Repository

```bash
git clone https://github.com/yogeshwenwu/GPU_Training.git
cd GPU_Training
```

If you want the OpenCL-enabled branch:

```bash
git checkout dev
```

---

## Running CPU Programs

### 1. Vector Addition (CPU)
File:
```bash
cpu/VectorAddition/vectorAddition.cpp
```

Compile:
```bash
g++ -O3 -mavx2 -std=c++17 cpu/VectorAddition/vectorAddition.cpp -o vectorAddition_cpu
```

Run:
```bash
./vectorAddition_cpu
```

### 2. Matrix Multiplication (CPU)
File:
```bash
cpu/MatrixMultiplication/MatrixMultiplication.cpp
```

Compile:
```bash
g++ -O3 -mavx2 -std=c++17 cpu/MatrixMultiplication/MatrixMultiplication.cpp -o matmul_cpu
```

Run:
```bash
./matmul_cpu
```

### 3. Matrix Multiplication AVX
File:
```bash
cpu/MatrixMultiplication/MatMulAVX.cpp
```

Compile:
```bash
g++ -O3 -mavx2 -std=c++17 cpu/MatrixMultiplication/MatMulAVX.cpp -o matmul_avx
```

Run:
```bash
./matmul_avx
```

### 4. Matrix Transpose
File:
```bash
cpu/MatrixTranspose/MatrixTranspose.cpp
```

Compile:
```bash
g++ -O3 -std=c++17 cpu/MatrixTranspose/MatrixTranspose.cpp -o matrix_transpose
```

Run:
```bash
./matrix_transpose
```

---

## Running CUDA Programs

### 1. Vector Addition on GPU
File:
```bash
gpu/Vector_Addition/vectorAddition.cu
```

Compile:
```bash
nvcc gpu/Vector_Addition/vectorAddition.cu -o vectorAddition_cuda
```

Run:
```bash
./vectorAddition_cuda
```

### 2. Matrix Multiplication (Native CUDA)
File:
```bash
gpu/Matrix Multiplication/matMulNative.cu
```

Compile:
```bash
nvcc "gpu/Matrix Multiplication/matMulNative.cu" -o matMulNative
```

Run:
```bash
./matMulNative
```

### 3. Shared Memory Matrix Multiplication
File:
```bash
gpu/Matrix Multiplication/matMulShared.cu
```

Compile:
```bash
nvcc "gpu/Matrix Multiplication/matMulShared.cu" -o matMulShared
```

Run:
```bash
./matMulShared
```

### 4. Tiled Matrix Multiplication
File:
```bash
gpu/Matrix Multiplication/matMulTile.cu
```

Compile:
```bash
nvcc "gpu/Matrix Multiplication/matMulTile.cu" -o matMulTile
```

Run:
```bash
./matMulTile
```

### 5. Warp-based Matrix Multiplication
File:
```bash
gpu/Matrix Multiplication/matmulWarp.cu
```

Compile:
```bash
nvcc "gpu/Matrix Multiplication/matmulWarp.cu" -o matmulWarp
```

Run:
```bash
./matmulWarp
```

---

## Running OpenCL Programs

OpenCL examples are stored in:
```bash
OpenCL/
```

These examples differ depending on the project structure, but the general compile workflow is:

### Example pattern
```bash
gcc -I/usr/include -o opencl_vectoradd OpenCL/vectorAddition/main.c -lOpenCL
```

or, if the file is a C++ OpenCL program:

```bash
g++ -std=c++17 -o opencl_vectoradd OpenCL/vectorAddition/main.cpp -lOpenCL
```

Run:
```bash
./opencl_vectoradd
```

For matrix multiplication:
```bash
g++ -std=c++17 -o opencl_matmul OpenCL/MatMul/main.cpp -lOpenCL
./opencl_matmul
```

The exact file names and build steps depend on the contents inside each OpenCL folder, so inspect the source before building.

---

## Build Commands Summary

### CPU
```bash
g++ -O3 -mavx2 -std=c++17 cpu/VectorAddition/vectorAddition.cpp -o vectorAddition_cpu
./vectorAddition_cpu
```

### CUDA
```bash
nvcc gpu/Vector_Addition/vectorAddition.cu -o vectorAddition_cuda
./vectorAddition_cuda
```

### OpenCL
```bash
g++ -std=c++17 -o opencl_example OpenCL/vectorAddition/main.cpp -lOpenCL
./opencl_example
```

---

## Notes

- These examples are mainly educational and benchmark-oriented.
- Several files compare naive implementations vs optimized versions.
- CUDA programs require an NVIDIA GPU and CUDA runtime.
- OpenCL examples are expected on the `dev` branch.
- Some sample programs measure execution time and print validation results.

---

## Learning Path

A recommended order:
1. Run `cpu/VectorAddition/vectorAddition.cpp`
2. Compare with `gpu/Vector_Addition/vectorAddition.cu`
3. Explore `gpu/Matrix Multiplication/`
4. Study `OpenCL/` for portable accelerator programming

---

## License

This repository does not currently include an explicit license file. Please check with the repository owner before using, modifying, or redistributing code for public or commercial purposes.

---

## Goal of the Project

This project is designed to help developers and students:
- understand CPU vs GPU execution
- study parallel compute patterns
- explore SIMD, CUDA, and OpenCL programming
- benchmark performance and evaluate optimization trade-offs

---

Built for learning GPU and parallel programming fundamentals through practical code examples.
