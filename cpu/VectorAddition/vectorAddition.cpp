#include <iostream>
#include <immintrin.h>
#include <vector>
#include <chrono>

// #pragma GCC optimize("O2")
// #pragma GCC target("sse,sse2,avx,avx2")

using namespace std;

// Scalar Version of Vector Addition
void vectorAdditionScalar(const vector<float> &a, const vector<float> &b, vector<float> &c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

// Scalar Version of Vector Addition with Loop Unrolling
void vectorAdditionLoopUnroll(const vector<float> &a, const vector<float> &b, vector<float> &c, int n)
{
    int remainder = n % 4;
    int limit = n - remainder;

    for (int i = 0; i < limit; i += 4)
    {
        c[i] = a[i] + b[i];
        c[i + 1] = a[i + 1] + b[i + 1];
        c[i + 2] = a[i + 2] + b[i + 2];
        c[i + 3] = a[i + 3] + b[i + 3];
    }
    for (int i = limit; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

// Intrinsic Version of Vector Addition (SSE)
void vectorAdditionSSE(const vector<float> &a, const vector<float> &b, vector<float> &c, int n)
{
    int remainder = n % 4;
    int limit = n - remainder;

    for (int i = 0; i < limit; i += 4)
    {
        __m128 vecA = _mm_loadu_ps(&a[i]);
        __m128 vecB = _mm_loadu_ps(&b[i]);
        __m128 vecC = _mm_add_ps(vecA, vecB);
        _mm_storeu_ps(&c[i], vecC);
    }

    for (int i = limit; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

// Intrinsic Version of Vector Addition (AVX)
void vectorAdditionAVX(const vector<float> &a, const vector<float> &b, vector<float> &c, int n)
{
    int remainder = n % 8;
    int limit = n - remainder;

    for (int i = 0; i < limit; i += 8)
    {
        __m256 vecAA = _mm256_loadu_ps(&a[i]);
        __m256 vecBB = _mm256_loadu_ps(&b[i]);
        __m256 vecCC = _mm256_add_ps(vecAA, vecBB);
        _mm256_storeu_ps(&c[i], vecCC);
    }

    for (int i = limit; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

bool verify(const vector<float> &a, const vector<float> &b, vector<float> &c, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (c[i] != a[i] + b[i])
        {
            cout << "Error at index " << i << ": "
                 << c[i] << " != " << a[i] + b[i] << endl;
            return false;
        }
    }
    fill(c.begin(), c.end(), 0.0f);
    return true;
}

int main()
{
    int N;
    // cin >> N;
    N = 87654321;
    int warmup_iter = 10;

    // Initialize vectors
    vector<float> a(N), b(N), c(N);

    for (int i = 0; i < N; i++)
    {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    // Scalar Version
    // warmup run
    for (int i = 0; i < warmup_iter; i++)
    {
        vectorAdditionScalar(a, b, c, N);
    }

    // actual run
    auto scalar_start = chrono::high_resolution_clock::now();
    vectorAdditionScalar(a, b, c, N);
    auto scalar_end = chrono::high_resolution_clock::now();

    auto scalar_duration = chrono::duration_cast<chrono::nanoseconds>(scalar_end - scalar_start);
    cout << "Scalar Version Time: " << scalar_duration.count() << " ns" << endl;
    if (!verify(a, b, c, N))
        return -1;

    // Loop Unroll Version
    // warmup run
    for (int i = 0; i < warmup_iter; i++)
    {
        vectorAdditionLoopUnroll(a, b, c, N);
    }
    // actual run
    auto loop_unroll_start = chrono::high_resolution_clock::now();
    vectorAdditionLoopUnroll(a, b, c, N);
    auto loop_unroll_end = chrono::high_resolution_clock::now();

    if (!verify(a, b, c, N))
        return -1;
    auto loop_unroll_duration = chrono::duration_cast<chrono::nanoseconds>(loop_unroll_end - loop_unroll_start);
    cout << "Loop Unroll Version Time: " << loop_unroll_duration.count() << " ns" << endl;

    // SIMD programming
    // SSE Version
    // warmup run
    for (int i = 0; i < warmup_iter; i++)
    {
        vectorAdditionSSE(a, b, c, N);
    }
    // actual run
    auto sse_start = chrono::high_resolution_clock::now();
    vectorAdditionSSE(a, b, c, N);
    auto sse_end = chrono::high_resolution_clock::now();

    if (!verify(a, b, c, N))
        return -1;
    auto sse_duration = chrono::duration_cast<chrono::nanoseconds>(sse_end - sse_start);
    cout << "SSE Version Time: " << sse_duration.count() << " ns" << endl;

    // AVX version
    // warmup run
    for (int i = 0; i < warmup_iter; i++)
    {
        vectorAdditionAVX(a, b, c, N);
    }
    // actual run
    auto avx_start = chrono::high_resolution_clock::now();
    vectorAdditionAVX(a, b, c, N);
    auto avx_end = chrono::high_resolution_clock::now();

    if (!verify(a, b, c, N))
        return -1;
    auto avx_duration = chrono::duration_cast<chrono::nanoseconds>(avx_end - avx_start);
    cout << "AVX Version Time: " << avx_duration.count() << " ns" << endl;
}
