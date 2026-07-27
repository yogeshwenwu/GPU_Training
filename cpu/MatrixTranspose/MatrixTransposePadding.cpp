#include <iostream>
#include <xmmintrin.h>
#include <immintrin.h>
#include <vector>
#include <chrono>
#include <string.h>

#define M 807
#define N 807
#define LOG 0

using namespace std;

// constexpr int paddedM = (M + 3) & ~3;
// constexpr int paddedN = (N + 3) & ~3;
constexpr int paddedM = (M + 7) & ~7; // ((M + 8 - 1) / 8) * 8
constexpr int paddedN = (N + 7) & ~7;

// Scalar version
void MatrixTranspose(float matrixA[][paddedN], float matrixB[][paddedM])
{
    for (int i = 0; i < paddedM; i++)
    {
        for (int j = 0; j < paddedN; j++)
        {
            matrixB[j][i] = matrixA[i][j];
        }
    }
}

// SSE version with _MM_Transpose4_PS()
void MatrixTransposeSSE(float matrixA[][paddedN], float matrixB[][paddedM]) 
{
    // considering squared matrix which is divisible by 4 (best case)
    for(int i = 0; i < paddedM; i += 4) 
    {
        for(int j = 0; j < paddedN; j+=4)
        {
            // load matrix to 128bit registers
            __m128 regA = _mm_loadu_ps(&matrixA[i][j]); // equivalent to _mm_load_ps(&matrixA[0][0]);
            __m128 regB = _mm_loadu_ps(&matrixA[i+1][j]); // equivalent to _mm_load_ps(&matrixA[1][0]);
            __m128 regC = _mm_loadu_ps(&matrixA[i+2][j]); // equivalent to _mm_load_ps(&matrixA[2][0]);
            __m128 regD = _mm_loadu_ps(&matrixA[i+3][j]); // equivalent to _mm_load_ps(&matrixA[3][0]);

            // perform matrix transpose
            _MM_TRANSPOSE4_PS(regA, regB, regC, regD);

            //store back to the matrixB
            _mm_storeu_ps(&matrixB[j][i], regA);
            _mm_storeu_ps(&matrixB[j+1][i], regB);
            _mm_storeu_ps(&matrixB[j+2][i], regC);
            _mm_storeu_ps(&matrixB[j+3][i], regD);
        }
    }
}

// SSE version with Padding
void MatrixTransposeSSEPadding(float matrixA[][paddedN], float matrixB[][paddedM]) 
{
    // considering squared matrix which is divisible by 4 (best case)
    for(int i = 0; i < paddedM; i += 4) 
    {
        for(int j = 0; j < paddedN; j+=4)
        {
            // load matrix to 128bit registers
            __m128 regA = _mm_loadu_ps(&matrixA[i][j]); // equivalent to _mm_load_ps(&matrixA[0][0]);
            __m128 regB = _mm_loadu_ps(&matrixA[i+1][j]); // equivalent to _mm_load_ps(&matrixA[1][0]);
            __m128 regC = _mm_loadu_ps(&matrixA[i+2][j]); // equivalent to _mm_load_ps(&matrixA[2][0]);
            __m128 regD = _mm_loadu_ps(&matrixA[i+3][j]); // equivalent to _mm_load_ps(&matrixA[3][0]);

            // perform matrix transpose
            _MM_TRANSPOSE4_PS(regA, regB, regC, regD);

            //store back to the matrixB
            _mm_storeu_ps(&matrixB[j][i], regA);
            _mm_storeu_ps(&matrixB[j+1][i], regB);
            _mm_storeu_ps(&matrixB[j+2][i], regC);
            _mm_storeu_ps(&matrixB[j+3][i], regD);
        }
    }
}

// Native Implementation of SSE
void MatrixTransposeSSENative(float matrixA[][paddedN], float matrixB[][paddedM]) 
{
    // considering squared matrix which is divisible by 4 (best case)
    for(int i = 0; i < paddedM; i += 4) 
    {
        for(int j = 0; j < paddedN; j+=4)
        {
            __m128 t0, t1, t2, t3;
            // load matrix to 128bit registers
            __m128 regA = _mm_loadu_ps(&matrixA[i][j]); // equivalent to _mm_load_ps(&matrixA[0][0]);
            __m128 regB = _mm_loadu_ps(&matrixA[i+1][j]); // equivalent to _mm_load_ps(&matrixA[1][0]);
            __m128 regC = _mm_loadu_ps(&matrixA[i+2][j]); // equivalent to _mm_load_ps(&matrixA[2][0]);
            __m128 regD = _mm_loadu_ps(&matrixA[i+3][j]); // equivalent to _mm_load_ps(&matrixA[3][0]);

            // perform matrix transpose
            // Step1: Unpack
            t0 = _mm_unpacklo_ps(regA, regB);
            t1 = _mm_unpackhi_ps(regA, regB);
            t2 = _mm_unpacklo_ps(regC, regD);
            t3 = _mm_unpackhi_ps(regC, regD);

            // Step2: move
            regA = _mm_movelh_ps(t0, t2);
            regB = _mm_movehl_ps(t2, t0);
            regC = _mm_movelh_ps(t1, t3);
            regD = _mm_movehl_ps(t3, t1);

            // store back to the matrixB
            _mm_storeu_ps(&matrixB[j][i], regA);
            _mm_storeu_ps(&matrixB[j+1][i], regB);
            _mm_storeu_ps(&matrixB[j+2][i], regC);
            _mm_storeu_ps(&matrixB[j+3][i], regD);
        }
    }
}


void MatrixTransposeAVX(float matrixA[][paddedN], float matrixB[][paddedM])
{
//    #pragma omp parallel for collapse(2)
    for(int i=0; i<paddedM; i+=8)
    {
        for(int j=0; j<paddedN; j+=8)
        {
            // load the matrix
            __m256 reg0 = _mm256_loadu_ps(&matrixA[i][j]);
            __m256 reg1 = _mm256_loadu_ps(&matrixA[i+1][j]);
            __m256 reg2 = _mm256_loadu_ps(&matrixA[i+2][j]);
            __m256 reg3 = _mm256_loadu_ps(&matrixA[i+3][j]);
            __m256 reg4 = _mm256_loadu_ps(&matrixA[i+4][j]);
            __m256 reg5 = _mm256_loadu_ps(&matrixA[i+5][j]);
            __m256 reg6 = _mm256_loadu_ps(&matrixA[i+6][j]);
            __m256 reg7 = _mm256_loadu_ps(&matrixA[i+7][j]);

            // transpose matrix
            __m256 t0, t1, t2, t3, t4, t5, t6, t7;
        
            // Step1: unpack
            t0 = _mm256_unpacklo_ps(reg0, reg1);
            t1 = _mm256_unpackhi_ps(reg0, reg1);
            t2 = _mm256_unpacklo_ps(reg2, reg3);
            t3 = _mm256_unpackhi_ps(reg2, reg3);
            t4 = _mm256_unpacklo_ps(reg4, reg5);
            t5 = _mm256_unpackhi_ps(reg4, reg5);
            t6 = _mm256_unpacklo_ps(reg6, reg7);
            t7 = _mm256_unpackhi_ps(reg6, reg7);

            // Step2: Shuffle
            __m256 s0,s1,s2,s3,s4,s5,s6,s7;
            s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1,0,1,0));
            s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3,2,3,2));
            s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1,0,1,0));
            s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3,2,3,2));
            s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1,0,1,0));
            s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3,2,3,2));
            s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1,0,1,0));
            s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3,2,3,2));

            // Step3: Permute
            t0 = _mm256_permute2f128_ps(s0, s4, 0x20);
            t1 = _mm256_permute2f128_ps(s1, s5, 0x20);
            t2 = _mm256_permute2f128_ps(s2, s6, 0x20);
            t3 = _mm256_permute2f128_ps(s3, s7, 0x20);
            t4 = _mm256_permute2f128_ps(s0, s4, 0x31);
            t5 = _mm256_permute2f128_ps(s1, s5, 0x31);
            t6 = _mm256_permute2f128_ps(s2, s6, 0x31);
            t7 = _mm256_permute2f128_ps(s3, s7, 0x31);

            // store back in array
            _mm256_storeu_ps(&matrixB[j][i], t0);
            _mm256_storeu_ps(&matrixB[j+1][i], t1);
            _mm256_storeu_ps(&matrixB[j+2][i], t2);
            _mm256_storeu_ps(&matrixB[j+3][i], t3);
            _mm256_storeu_ps(&matrixB[j+4][i], t4);
            _mm256_storeu_ps(&matrixB[j+5][i], t5);
            _mm256_storeu_ps(&matrixB[j+6][i], t6);
            _mm256_storeu_ps(&matrixB[j+7][i], t7);
        }
    }
}

bool verify(float matrixScalar[][paddedM], float matrixSIMD[][paddedM]){
    for (int i=0; i<N; i++)
    {
        for(int j=0; j<M; j++){
            if(matrixScalar[i][j] != matrixSIMD[i][j]){
                cout << "Mismatch in value: Expected = " << matrixScalar[i][j] << " but got = " << matrixSIMD[i][j] << endl;
                return false;
            }
        }
    }
    cout << "Test Passed" << endl;
    return true;
}

int main()
{
    cout << "Padded M: " << paddedM << endl << " Padded N: " << paddedN << endl;

    float matrixA[paddedM][paddedN] = {0};
    float matrixB[paddedN][paddedM] = {0};
    float matrixC[paddedN][paddedM] = {0};

    int warmup_iter = 100;

    cout << "Matrix A: Before Transpose" << endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrixA[i][j] = (N * i) + j;
            #if LOG
                cout << " " << matrixA[i][j] << " ";
            #endif
        }
        #if LOG
            cout << endl;
        #endif
    }

    // scalar version call
    // // warmup run
    for (int i=0; i < warmup_iter; i++) {
        MatrixTranspose(matrixA, matrixB);
    }

    // actual run
    auto scalar_start = chrono::high_resolution_clock::now();
    for(int i=0; i<100;i++)
    {
        MatrixTranspose(matrixA, matrixB);
    }
    // MatrixTranspose(matrixA, matrixB);
    auto scalar_end = chrono::high_resolution_clock::now();

    auto scalar_duration = chrono::duration_cast<chrono::nanoseconds>(scalar_end - scalar_start);
    cout << "Scalar Version Time: " << scalar_duration.count() / 100 << " ns" << endl;

    #if LOG
    cout << "Matrix B: After Transpose" << endl;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < M; j++)
            {
                cout << " " << matrixB[i][j] << " ";
            }
            cout << endl;
        }
    #endif

    // SSE verison call
    // warmup run
    for (int i=0; i < warmup_iter; i++) {
        MatrixTransposeSSE(matrixA, matrixC);
    }

    // actual run
    auto sse_start = chrono::high_resolution_clock::now();
    for(int i=0; i<100;i++)
    {
        MatrixTransposeSSE(matrixA, matrixC);
    }
    // MatrixTransposeSSE(matrixA, matrixC);
    auto sse_end = chrono::high_resolution_clock::now();

    auto sse_duration = chrono::duration_cast<chrono::nanoseconds>(sse_end - sse_start);
    cout << "SSE Version Time: " << sse_duration.count() / 100 << " ns" << endl;

    #if LOG
        cout << "Matrix C: After Transpose" << endl;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < M; j++)
            {
                cout << " " << matrixC[i][j] << " ";
            }
            cout << endl;
        }
    #endif

    if (!verify(matrixB, matrixC)) 
    {
        return -1;
    }
    memset(matrixC, 0, sizeof(matrixC));

    // SSE Native verison call
    // // warmup run
    for (int i=0; i < warmup_iter; i++) {
        MatrixTransposeSSENative(matrixA, matrixC);
    }

    // actual run
    auto ssen_start = chrono::high_resolution_clock::now();
    for(int i=0; i<100;i++)
    {
        MatrixTransposeSSENative(matrixA, matrixC);
    }
    // MatrixTransposeSSENative(matrixA, matrixC);
    auto ssen_end = chrono::high_resolution_clock::now();

    auto ssen_duration = chrono::duration_cast<chrono::nanoseconds>(ssen_end - ssen_start);
    cout << "SSEN Version Time: " << ssen_duration.count() / 100 << " ns" << endl;

    #if LOG
        cout << "Matrix C: After Transpose" << endl;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < M; j++)
            {
                cout << " " << matrixC[i][j] << " ";
            }
            cout << endl;
        }
    #endif

    if (!verify(matrixB, matrixC)) 
    {
        return -1;
    }
    memset(matrixC, 0, sizeof(matrixC));

    // AVX version call
    // warmup run
    for (int i=0; i < warmup_iter; i++) {
        MatrixTransposeAVX(matrixA, matrixC);
    }

    // actual run
    auto avx_start = chrono::high_resolution_clock::now();
    for(int i=0; i<100;i++)
    {
        MatrixTransposeAVX(matrixA, matrixC);
    }
    auto avx_end = chrono::high_resolution_clock::now();

    auto avx_duration = chrono::duration_cast<chrono::nanoseconds>(avx_end - avx_start);
    cout << "AVX Version Time: " << avx_duration.count() / 100 << " ns" << endl;

    #if LOG
        cout << "Matrix C: After Transpose" << endl;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < M; j++)
            {
                cout << " " << matrixC[i][j] << " ";
            }
            cout << endl;
        }
    #endif

    if (!verify(matrixB, matrixC)) 
    {
        return -1;
    }
    memset(matrixC, 0, sizeof(matrixC));

}