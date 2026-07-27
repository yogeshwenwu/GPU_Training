#include <iostream>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <string.h>
#include <cmath>
#include <chrono>

constexpr int M = 150;
constexpr int N = 150;
constexpr int K = 150;

int warmup_iter = 10;

constexpr int paddedM = ((M + 3) / 4) * 4;
constexpr int paddedN = ((N + 3) / 4) * 4;
constexpr int paddedK = ((K + 3) / 4) * 4;

using namespace std;

void MatMulScalar(float matRes[M][K], float matA[][N], float matB[][K])
{
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < M; j++)
        {
            float tmp = 0.0f;
            for (int k = 0; k < N; k++)
            {
                tmp = tmp + (matA[j][k] * matB[k][i]);
            }
            matRes[j][i] = tmp;
        }
    }
}

void MatrixTransposeSSE(float matTps[K][N], float matB[N][K])
{
    int rowLimit = N - (N % 4); // 4
    int colLimit = K - (K % 4); // 4
    // considering squared matrix which is divisible by 4 (best case)
    for (int i = 0; i < rowLimit; i += 4)
    {
        for (int j = 0; j < colLimit; j += 4)
        {
            // load matrix to 128bit registers
            __m128 regA = _mm_loadu_ps(&matB[i][j]);     // equivalent to _mm_load_ps(&matrixA[0][0]);
            __m128 regB = _mm_loadu_ps(&matB[i + 1][j]); // equivalent to _mm_load_ps(&matrixA[1][0]);
            __m128 regC = _mm_loadu_ps(&matB[i + 2][j]); // equivalent to _mm_load_ps(&matrixA[2][0]);
            __m128 regD = _mm_loadu_ps(&matB[i + 3][j]); // equivalent to _mm_load_ps(&matrixA[3][0]);

            // perform matrix transpose
            _MM_TRANSPOSE4_PS(regA, regB, regC, regD);

            // store back to the matrixB
            _mm_storeu_ps(&matTps[j][i], regA);
            _mm_storeu_ps(&matTps[j + 1][i], regB);
            _mm_storeu_ps(&matTps[j + 2][i], regC);
            _mm_storeu_ps(&matTps[j + 3][i], regD);
        }
    }
    if (rowLimit < N || colLimit < K)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < K; j++)
            {
                if (i >= rowLimit || j >= colLimit)
                {
                    matTps[j][i] = matB[i][j];
                }
            }
        }
    }
}

void MatrixTransposeSSEPadded(float matTps[][paddedN], float matB[][paddedK])
{
    // considering squared matrix which is divisible by 4 (best case)
    for (int i = 0; i < paddedN; i += 4)
    {
        for (int j = 0; j < paddedK; j += 4)
        {
            // load matrix to 128bit registers
            __m128 regA = _mm_loadu_ps(&matB[i][j]);     // equivalent to _mm_load_ps(&matrixA[0][0]);
            __m128 regB = _mm_loadu_ps(&matB[i + 1][j]); // equivalent to _mm_load_ps(&matrixA[1][0]);
            __m128 regC = _mm_loadu_ps(&matB[i + 2][j]); // equivalent to _mm_load_ps(&matrixA[2][0]);
            __m128 regD = _mm_loadu_ps(&matB[i + 3][j]); // equivalent to _mm_load_ps(&matrixA[3][0]);

            // perform matrix transpose
            _MM_TRANSPOSE4_PS(regA, regB, regC, regD);

            // store back to the matrixB
            _mm_storeu_ps(&matTps[j][i], regA);
            _mm_storeu_ps(&matTps[j + 1][i], regB);
            _mm_storeu_ps(&matTps[j + 2][i], regC);
            _mm_storeu_ps(&matTps[j + 3][i], regD);
        }
    }
}

void MatMulSSE4x4(float matRes[M][K], float matA[M][N], float matB[N][K])
{
    // matB transpose
    float matTmp[K][N];

    __m128 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3, ma00, ma01, ma02, ma03, ma10, ma11, ma12, ma13;
    __m128 ma20, ma21, ma22, ma23, ma30, ma31, ma32, ma33;

    // transpose matrix B
    MatrixTransposeSSE(matTmp, matB);

    // load matrix A
    ra0 = _mm_loadu_ps(matA[0]);
    ra1 = _mm_loadu_ps(matA[1]);
    ra2 = _mm_loadu_ps(matA[2]);
    ra3 = _mm_loadu_ps(matA[3]);

    // load matrix B
    rb0 = _mm_loadu_ps(matTmp[0]);
    rb1 = _mm_loadu_ps(matTmp[1]);
    rb2 = _mm_loadu_ps(matTmp[2]);
    rb3 = _mm_loadu_ps(matTmp[3]);

    // multiply
    ma00 = _mm_mul_ps(ra0, rb0);
    ma01 = _mm_mul_ps(ra0, rb1);
    ma02 = _mm_mul_ps(ra0, rb2);
    ma03 = _mm_mul_ps(ra0, rb3);

    ma10 = _mm_mul_ps(ra1, rb0);
    ma11 = _mm_mul_ps(ra1, rb1);
    ma12 = _mm_mul_ps(ra1, rb2);
    ma13 = _mm_mul_ps(ra1, rb3);

    ma20 = _mm_mul_ps(ra2, rb0);
    ma21 = _mm_mul_ps(ra2, rb1);
    ma22 = _mm_mul_ps(ra2, rb2);
    ma23 = _mm_mul_ps(ra2, rb3);

    ma30 = _mm_mul_ps(ra3, rb0);
    ma31 = _mm_mul_ps(ra3, rb1);
    ma32 = _mm_mul_ps(ra3, rb2);
    ma33 = _mm_mul_ps(ra3, rb3);

    // horizontal add
    ma00 = _mm_hadd_ps(ma00, ma00); // [1,2|3,4] [1,2|3,4] -> [3,7 | 3,7]
    ma00 = _mm_hadd_ps(ma00, ma00); // [3,7|3,7] [3,7|3,7] -> [10,10 | 10,10]
    ma01 = _mm_hadd_ps(ma01, ma01);
    ma01 = _mm_hadd_ps(ma01, ma01);
    ma02 = _mm_hadd_ps(ma02, ma02);
    ma02 = _mm_hadd_ps(ma02, ma02);
    ma03 = _mm_hadd_ps(ma03, ma03);
    ma03 = _mm_hadd_ps(ma03, ma03);

    ma10 = _mm_hadd_ps(ma10, ma10);
    ma10 = _mm_hadd_ps(ma10, ma10);
    ma11 = _mm_hadd_ps(ma11, ma11);
    ma11 = _mm_hadd_ps(ma11, ma11);
    ma12 = _mm_hadd_ps(ma12, ma12);
    ma12 = _mm_hadd_ps(ma12, ma12);
    ma13 = _mm_hadd_ps(ma13, ma13);
    ma13 = _mm_hadd_ps(ma13, ma13);

    ma20 = _mm_hadd_ps(ma20, ma20);
    ma20 = _mm_hadd_ps(ma20, ma20);
    ma21 = _mm_hadd_ps(ma21, ma21);
    ma21 = _mm_hadd_ps(ma21, ma21);
    ma22 = _mm_hadd_ps(ma22, ma22);
    ma22 = _mm_hadd_ps(ma22, ma22);
    ma23 = _mm_hadd_ps(ma23, ma23);
    ma23 = _mm_hadd_ps(ma23, ma23);

    ma30 = _mm_hadd_ps(ma30, ma30);
    ma30 = _mm_hadd_ps(ma30, ma30);
    ma31 = _mm_hadd_ps(ma31, ma31);
    ma31 = _mm_hadd_ps(ma31, ma31);
    ma32 = _mm_hadd_ps(ma32, ma32);
    ma32 = _mm_hadd_ps(ma32, ma32);
    ma33 = _mm_hadd_ps(ma33, ma33);
    ma33 = _mm_hadd_ps(ma33, ma33);

    // arrange
    ra0 = _mm_unpacklo_ps(ma00, ma01);
    rb0 = _mm_unpacklo_ps(ma02, ma03);
    ma00 = _mm_movelh_ps(ra0, rb0);

    ra1 = _mm_unpacklo_ps(ma10, ma11);
    rb1 = _mm_unpacklo_ps(ma12, ma13);
    ma10 = _mm_movelh_ps(ra1, rb1);

    ra2 = _mm_unpacklo_ps(ma20, ma21);
    rb2 = _mm_unpacklo_ps(ma22, ma23);
    ma20 = _mm_movelh_ps(ra2, rb2);

    ra3 = _mm_unpacklo_ps(ma30, ma31);
    rb3 = _mm_unpacklo_ps(ma32, ma33);
    ma30 = _mm_movelh_ps(ra3, rb3);

    // store
    _mm_storeu_ps(matRes[0], ma00);
    _mm_storeu_ps(matRes[1], ma10);
    _mm_storeu_ps(matRes[2], ma20);
    _mm_storeu_ps(matRes[3], ma30);
}

void MatMulSSE(float matRes[M][K], float matA[M][N], float matB[N][K])
{
    // matB transpose
    float matTmp[K][N];

    __m128 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3, ma00, ma01, ma02, ma03, ma10, ma11, ma12, ma13;
    __m128 ma20, ma21, ma22, ma23, ma30, ma31, ma32, ma33;

    // transpose matrix B
    MatrixTransposeSSE(matTmp, matB);

    constexpr int mLimit = M - (M % 4);
    constexpr int nLimit = N - (N % 4);
    constexpr int kLimit = K - (K % 4);

    // for (int k = 0; k < K; k += 4)
    // {
    //     for (int m = 0; m < M; m += 4)
    //     {
    for (int m = 0; m < mLimit; m += 4)
    {
        for (int k = 0; k < kLimit; k += 4)
        {
            __m128 sum0 = _mm_setzero_ps();
            __m128 sum1 = _mm_setzero_ps();
            __m128 sum2 = _mm_setzero_ps();
            __m128 sum3 = _mm_setzero_ps();
            for (int n = 0; n < nLimit; n += 4)
            {
                // load matrix A
                ra0 = _mm_loadu_ps(&matA[m][n]);
                ra1 = _mm_loadu_ps(&matA[m + 1][n]);
                ra2 = _mm_loadu_ps(&matA[m + 2][n]);
                ra3 = _mm_loadu_ps(&matA[m + 3][n]);

                // load matrix B
                rb0 = _mm_loadu_ps(&matTmp[k][n]);
                rb1 = _mm_loadu_ps(&matTmp[k + 1][n]);
                rb2 = _mm_loadu_ps(&matTmp[k + 2][n]);
                rb3 = _mm_loadu_ps(&matTmp[k + 3][n]);

                // multiply
                ma00 = _mm_mul_ps(ra0, rb0);
                ma01 = _mm_mul_ps(ra0, rb1);
                ma02 = _mm_mul_ps(ra0, rb2);
                ma03 = _mm_mul_ps(ra0, rb3);

                ma10 = _mm_mul_ps(ra1, rb0);
                ma11 = _mm_mul_ps(ra1, rb1);
                ma12 = _mm_mul_ps(ra1, rb2);
                ma13 = _mm_mul_ps(ra1, rb3);

                ma20 = _mm_mul_ps(ra2, rb0);
                ma21 = _mm_mul_ps(ra2, rb1);
                ma22 = _mm_mul_ps(ra2, rb2);
                ma23 = _mm_mul_ps(ra2, rb3);

                ma30 = _mm_mul_ps(ra3, rb0);
                ma31 = _mm_mul_ps(ra3, rb1);
                ma32 = _mm_mul_ps(ra3, rb2);
                ma33 = _mm_mul_ps(ra3, rb3);

                // horizontal add
                ma00 = _mm_hadd_ps(ma00, ma00); // [1,2|3,4] [1,2|3,4] -> [3,7 | 3,7]
                ma00 = _mm_hadd_ps(ma00, ma00); // [3,7|3,7] [3,7|3,7] -> [10,10 | 10,10]
                ma01 = _mm_hadd_ps(ma01, ma01);
                ma01 = _mm_hadd_ps(ma01, ma01);
                ma02 = _mm_hadd_ps(ma02, ma02);
                ma02 = _mm_hadd_ps(ma02, ma02);
                ma03 = _mm_hadd_ps(ma03, ma03);
                ma03 = _mm_hadd_ps(ma03, ma03);

                ma10 = _mm_hadd_ps(ma10, ma10);
                ma10 = _mm_hadd_ps(ma10, ma10);
                ma11 = _mm_hadd_ps(ma11, ma11);
                ma11 = _mm_hadd_ps(ma11, ma11);
                ma12 = _mm_hadd_ps(ma12, ma12);
                ma12 = _mm_hadd_ps(ma12, ma12);
                ma13 = _mm_hadd_ps(ma13, ma13);
                ma13 = _mm_hadd_ps(ma13, ma13);

                ma20 = _mm_hadd_ps(ma20, ma20);
                ma20 = _mm_hadd_ps(ma20, ma20);
                ma21 = _mm_hadd_ps(ma21, ma21);
                ma21 = _mm_hadd_ps(ma21, ma21);
                ma22 = _mm_hadd_ps(ma22, ma22);
                ma22 = _mm_hadd_ps(ma22, ma22);
                ma23 = _mm_hadd_ps(ma23, ma23);
                ma23 = _mm_hadd_ps(ma23, ma23);

                ma30 = _mm_hadd_ps(ma30, ma30);
                ma30 = _mm_hadd_ps(ma30, ma30);
                ma31 = _mm_hadd_ps(ma31, ma31);
                ma31 = _mm_hadd_ps(ma31, ma31);
                ma32 = _mm_hadd_ps(ma32, ma32);
                ma32 = _mm_hadd_ps(ma32, ma32);
                ma33 = _mm_hadd_ps(ma33, ma33);
                ma33 = _mm_hadd_ps(ma33, ma33);

                // arrange
                ra0 = _mm_unpacklo_ps(ma00, ma01);
                rb0 = _mm_unpacklo_ps(ma02, ma03);
                ma00 = _mm_movelh_ps(ra0, rb0);

                ra1 = _mm_unpacklo_ps(ma10, ma11);
                rb1 = _mm_unpacklo_ps(ma12, ma13);
                ma10 = _mm_movelh_ps(ra1, rb1);

                ra2 = _mm_unpacklo_ps(ma20, ma21);
                rb2 = _mm_unpacklo_ps(ma22, ma23);
                ma20 = _mm_movelh_ps(ra2, rb2);

                ra3 = _mm_unpacklo_ps(ma30, ma31);
                rb3 = _mm_unpacklo_ps(ma32, ma33);
                ma30 = _mm_movelh_ps(ra3, rb3);

                sum0 = _mm_add_ps(sum0, ma00);
                sum1 = _mm_add_ps(sum1, ma10);
                sum2 = _mm_add_ps(sum2, ma20);
                sum3 = _mm_add_ps(sum3, ma30);
            }

            // store
            _mm_storeu_ps(&matRes[m][k], sum0);
            _mm_storeu_ps(&matRes[m + 1][k], sum1);
            _mm_storeu_ps(&matRes[m + 2][k], sum2);
            _mm_storeu_ps(&matRes[m + 3][k], sum3);
        }
    }

    if (mLimit < M || nLimit < N || kLimit < K)
    {
        for (int i = 0; i < M; i++)
        {
            for (int k = 0; k < K; k++)
            {
                float sum = 0.0f;
                int nStart = 0;

                if (i < mLimit && k < kLimit)
                {
                    sum = matRes[i][k];
                    nStart = nLimit;
                }

                for (int j = nStart; j < N; j++)
                {
                    sum = sum + matA[i][j] * matB[j][k];
                }

                matRes[i][k] = sum;
            }
        }
    }
}

void MatMulSSEPadded(float matRes[paddedM][paddedK], float matA[paddedM][paddedN], float matB[paddedN][paddedK])
{

    float matTmp[paddedK][paddedN];

    __m128 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3, ma00, ma01, ma02, ma03, ma10, ma11, ma12, ma13;
    __m128 ma20, ma21, ma22, ma23, ma30, ma31, ma32, ma33;

    // transpose matrix B
    MatrixTransposeSSEPadded(matTmp, matB);

    for (int m = 0; m < paddedM; m += 4)
    {
        for (int k = 0; k < paddedK; k += 4)
        {
            __m128 sum0 = _mm_setzero_ps();
            __m128 sum1 = _mm_setzero_ps();
            __m128 sum2 = _mm_setzero_ps();
            __m128 sum3 = _mm_setzero_ps();
            for (int n = 0; n < paddedN; n += 4)
            {
                // load matrix A
                ra0 = _mm_loadu_ps(&matA[m][n]);
                ra1 = _mm_loadu_ps(&matA[m + 1][n]);
                ra2 = _mm_loadu_ps(&matA[m + 2][n]);
                ra3 = _mm_loadu_ps(&matA[m + 3][n]);

                // load matrix B
                rb0 = _mm_loadu_ps(&matTmp[k][n]);
                rb1 = _mm_loadu_ps(&matTmp[k + 1][n]);
                rb2 = _mm_loadu_ps(&matTmp[k + 2][n]);
                rb3 = _mm_loadu_ps(&matTmp[k + 3][n]);

                // multiply
                ma00 = _mm_mul_ps(ra0, rb0);
                ma01 = _mm_mul_ps(ra0, rb1);
                ma02 = _mm_mul_ps(ra0, rb2);
                ma03 = _mm_mul_ps(ra0, rb3);

                ma10 = _mm_mul_ps(ra1, rb0);
                ma11 = _mm_mul_ps(ra1, rb1);
                ma12 = _mm_mul_ps(ra1, rb2);
                ma13 = _mm_mul_ps(ra1, rb3);

                ma20 = _mm_mul_ps(ra2, rb0);
                ma21 = _mm_mul_ps(ra2, rb1);
                ma22 = _mm_mul_ps(ra2, rb2);
                ma23 = _mm_mul_ps(ra2, rb3);

                ma30 = _mm_mul_ps(ra3, rb0);
                ma31 = _mm_mul_ps(ra3, rb1);
                ma32 = _mm_mul_ps(ra3, rb2);
                ma33 = _mm_mul_ps(ra3, rb3);

                // horizontal add
                ma00 = _mm_hadd_ps(ma00, ma00); // [1,2|3,4] [1,2|3,4] -> [3,7 | 3,7]
                ma00 = _mm_hadd_ps(ma00, ma00); // [3,7|3,7] [3,7|3,7] -> [10,10 | 10,10]
                ma01 = _mm_hadd_ps(ma01, ma01);
                ma01 = _mm_hadd_ps(ma01, ma01);
                ma02 = _mm_hadd_ps(ma02, ma02);
                ma02 = _mm_hadd_ps(ma02, ma02);
                ma03 = _mm_hadd_ps(ma03, ma03);
                ma03 = _mm_hadd_ps(ma03, ma03);

                ma10 = _mm_hadd_ps(ma10, ma10);
                ma10 = _mm_hadd_ps(ma10, ma10);
                ma11 = _mm_hadd_ps(ma11, ma11);
                ma11 = _mm_hadd_ps(ma11, ma11);
                ma12 = _mm_hadd_ps(ma12, ma12);
                ma12 = _mm_hadd_ps(ma12, ma12);
                ma13 = _mm_hadd_ps(ma13, ma13);
                ma13 = _mm_hadd_ps(ma13, ma13);

                ma20 = _mm_hadd_ps(ma20, ma20);
                ma20 = _mm_hadd_ps(ma20, ma20);
                ma21 = _mm_hadd_ps(ma21, ma21);
                ma21 = _mm_hadd_ps(ma21, ma21);
                ma22 = _mm_hadd_ps(ma22, ma22);
                ma22 = _mm_hadd_ps(ma22, ma22);
                ma23 = _mm_hadd_ps(ma23, ma23);
                ma23 = _mm_hadd_ps(ma23, ma23);

                ma30 = _mm_hadd_ps(ma30, ma30);
                ma30 = _mm_hadd_ps(ma30, ma30);
                ma31 = _mm_hadd_ps(ma31, ma31);
                ma31 = _mm_hadd_ps(ma31, ma31);
                ma32 = _mm_hadd_ps(ma32, ma32);
                ma32 = _mm_hadd_ps(ma32, ma32);
                ma33 = _mm_hadd_ps(ma33, ma33);
                ma33 = _mm_hadd_ps(ma33, ma33);

                // arrange
                ra0 = _mm_unpacklo_ps(ma00, ma01);
                rb0 = _mm_unpacklo_ps(ma02, ma03);
                ma00 = _mm_movelh_ps(ra0, rb0);

                ra1 = _mm_unpacklo_ps(ma10, ma11);
                rb1 = _mm_unpacklo_ps(ma12, ma13);
                ma10 = _mm_movelh_ps(ra1, rb1);

                ra2 = _mm_unpacklo_ps(ma20, ma21);
                rb2 = _mm_unpacklo_ps(ma22, ma23);
                ma20 = _mm_movelh_ps(ra2, rb2);

                ra3 = _mm_unpacklo_ps(ma30, ma31);
                rb3 = _mm_unpacklo_ps(ma32, ma33);
                ma30 = _mm_movelh_ps(ra3, rb3);

                sum0 = _mm_add_ps(sum0, ma00);
                sum1 = _mm_add_ps(sum1, ma10);
                sum2 = _mm_add_ps(sum2, ma20);
                sum3 = _mm_add_ps(sum3, ma30);
            }

            // store
            _mm_storeu_ps(&matRes[m][k], sum0);
            _mm_storeu_ps(&matRes[m + 1][k], sum1);
            _mm_storeu_ps(&matRes[m + 2][k], sum2);
            _mm_storeu_ps(&matRes[m + 3][k], sum3);
        }
    }
}

bool verify(float matScalar[M][K], float matVec[M][K])
{
    constexpr float absTolerance = 1e-3f;
    constexpr float relTolerance = 1e-5f;

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            float diff = fabsf(matScalar[i][j] - matVec[i][j]);
            float limit = absTolerance + relTolerance * fmaxf(fabsf(matScalar[i][j]), fabsf(matVec[i][j]));

            if (diff > limit)
            {
                cout << "Mismatch at i:" << i << " , j:" << j
                     << " ; Scalar:" << matScalar[i][j]
                     << ", Vector:" << matVec[i][j]
                     << ", Diff:" << diff
                     << ", Tolerance:" << limit << endl;
                return false;
            }
        }
    }
    return true;
}

int main()
{
    // initialize the matrix
    float matrixA[M][N];
    float matrixB[N][K];
    float matrixRes[M][K];
    float matrixResV[M][K];

    // matB transpose
    float matAPad[paddedM][paddedN] = {0};
    float matBPad[paddedN][paddedK] = {0};
    float matResPad[paddedM][paddedK] = {0};

    cout << "Matrix A:" << endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrixA[i][j] = (N * i) + j;
            matAPad[i][j] = (N * i) + j;
            cout << matrixA[i][j] << " ";
        }
        cout << endl;
    }

    cout << "\nMatrix B:" << endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < K; j++)
        {
            matrixB[i][j] = i + j;
            matBPad[i][j] = i + j;
            cout << matrixB[i][j] << " ";
        }
        cout << endl;
    }

    cout << "Running Scalar code" << endl;
    // scalar version call
    // // warmup run
    for (int i = 0; i < warmup_iter; i++)
    {
        MatMulScalar(matrixRes, matrixA, matrixB);
    }

    // actual run
    auto scalar_start = chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++)
    {
        MatMulScalar(matrixRes, matrixA, matrixB);
    }
    auto scalar_end = chrono::high_resolution_clock::now();

    auto scalar_duration = chrono::duration_cast<chrono::nanoseconds>(scalar_end - scalar_start);

    cout << "\nResultant Matrix:" << endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            cout << matrixRes[i][j] << " ";
        }
        cout << endl;
    }
    memset(matrixResV, 0, sizeof(matrixRes));

    // cout << "\nResultant Matrix: Cleaned" << endl;

    cout << "Running SSE code" << endl;
    // SSE version call
    // // warmup run
    for (int i = 0; i < warmup_iter; i++)
    {
        MatMulSSE(matrixResV, matrixA, matrixB);
    }

    // actual run
    auto sse_start = chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++)
    {
        MatMulSSE(matrixResV, matrixA, matrixB);
    }
    // MatrixTranspose(matrixA, matrixB);
    auto sse_end = chrono::high_resolution_clock::now();

    auto sse_duration = chrono::duration_cast<chrono::nanoseconds>(sse_end - sse_start);

    cout << "\nSSE Resultant Matrix:" << endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            cout << matrixResV[i][j] << " ";
        }
        cout << endl;
    }
    if (!verify(matrixRes, matrixResV))
    {
        return -1;
    }
    memset(matrixRes, 0, sizeof(matrixRes));

    cout << "Running SSE Padded code" << endl;
    // SSE Padded version call
    // // warmup run
    for (int i = 0; i < warmup_iter; i++)
    {
        MatMulSSEPadded(matResPad, matAPad, matBPad);
    }

    // actual run
    auto sse_pad_start = chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++)
    {
        MatMulSSEPadded(matResPad, matAPad, matBPad);
    }
    // MatrixTranspose(matrixA, matrixB);
    auto sse_pad_end = chrono::high_resolution_clock::now();

    auto sse_pad_duration = chrono::duration_cast<chrono::nanoseconds>(sse_pad_end - sse_pad_start);

    cout << "\nSSE Padded Resultant Matrix:" << endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            matrixRes[i][j] = matResPad[i][j];
            cout << matResPad[i][j] << " ";
        }
        cout << endl;
    }
    if (!verify(matrixRes, matrixResV))
    {
        return -1;
    }
    cout << "\nResult: " << endl;
    cout << "Scalar Version Time: " << scalar_duration.count() / 10 << " ns" << endl;
    cout << "SSE Version Time: " << sse_duration.count() / 10 << " ns" << endl;
    cout << "SSE padded Version Time: " << sse_pad_duration.count() / 10 << " ns" << endl;
}
