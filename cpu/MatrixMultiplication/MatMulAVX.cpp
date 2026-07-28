#include <iostream>
#include <immintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <string.h>
#include <chrono>
#include <cmath>

constexpr int M = 150;
constexpr int N = 150;
constexpr int K = 150;

int warmup_iter = 10;

constexpr int paddedM = ((M + 7) / 8) * 8;
constexpr int paddedN = ((N + 7) / 8) * 8;
constexpr int paddedK = ((K + 7) / 8) * 8;

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

void MatrixTransposeAVX(float matTps[K][N], float matB[N][K])
{
    int rowLimit = N - (N % 8);
    int colLimit = K - (K % 8);
    for (int i = 0; i < rowLimit; i += 8)
    {
        for (int j = 0; j < colLimit; j += 8)
        {
            // load the matrix
            __m256 reg0 = _mm256_loadu_ps(&matB[i][j]);
            __m256 reg1 = _mm256_loadu_ps(&matB[i + 1][j]);
            __m256 reg2 = _mm256_loadu_ps(&matB[i + 2][j]);
            __m256 reg3 = _mm256_loadu_ps(&matB[i + 3][j]);
            __m256 reg4 = _mm256_loadu_ps(&matB[i + 4][j]);
            __m256 reg5 = _mm256_loadu_ps(&matB[i + 5][j]);
            __m256 reg6 = _mm256_loadu_ps(&matB[i + 6][j]);
            __m256 reg7 = _mm256_loadu_ps(&matB[i + 7][j]);

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
            __m256 s0, s1, s2, s3, s4, s5, s6, s7;
            s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
            s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
            s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
            s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
            s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
            s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
            s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
            s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

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
            _mm256_storeu_ps(&matTps[j][i], t0);
            _mm256_storeu_ps(&matTps[j + 1][i], t1);
            _mm256_storeu_ps(&matTps[j + 2][i], t2);
            _mm256_storeu_ps(&matTps[j + 3][i], t3);
            _mm256_storeu_ps(&matTps[j + 4][i], t4);
            _mm256_storeu_ps(&matTps[j + 5][i], t5);
            _mm256_storeu_ps(&matTps[j + 6][i], t6);
            _mm256_storeu_ps(&matTps[j + 7][i], t7);
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

void MatrixTransposeAVXPadded(float matTps[][paddedN], float matB[][paddedK])
{
    for (int i = 0; i < paddedN; i += 8)
    {
        for (int j = 0; j < paddedK; j += 8)
        {
            // load the matrix
            __m256 reg0 = _mm256_loadu_ps(&matB[i][j]);
            __m256 reg1 = _mm256_loadu_ps(&matB[i + 1][j]);
            __m256 reg2 = _mm256_loadu_ps(&matB[i + 2][j]);
            __m256 reg3 = _mm256_loadu_ps(&matB[i + 3][j]);
            __m256 reg4 = _mm256_loadu_ps(&matB[i + 4][j]);
            __m256 reg5 = _mm256_loadu_ps(&matB[i + 5][j]);
            __m256 reg6 = _mm256_loadu_ps(&matB[i + 6][j]);
            __m256 reg7 = _mm256_loadu_ps(&matB[i + 7][j]);

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
            __m256 s0, s1, s2, s3, s4, s5, s6, s7;
            s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
            s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
            s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
            s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
            s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
            s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
            s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
            s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

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
            _mm256_storeu_ps(&matTps[j][i], t0);
            _mm256_storeu_ps(&matTps[j + 1][i], t1);
            _mm256_storeu_ps(&matTps[j + 2][i], t2);
            _mm256_storeu_ps(&matTps[j + 3][i], t3);
            _mm256_storeu_ps(&matTps[j + 4][i], t4);
            _mm256_storeu_ps(&matTps[j + 5][i], t5);
            _mm256_storeu_ps(&matTps[j + 6][i], t6);
            _mm256_storeu_ps(&matTps[j + 7][i], t7);
        }
    }
}

static inline float hsum(__m256 rv)
{
    __m256 t = _mm256_hadd_ps(rv, rv);       // [a0+a1, a2+a3, a0+a1, a2+a3 | a4+a5, a6+a7, a4+a5, a6+a7]
    t = _mm256_hadd_ps(t, t);                // [a0+a1+a2+a3, a0+..., ... | a4+..+a7, ...]
    __m128 lo = _mm256_castps256_ps128(t);   // a0+a1+a2+a3
    __m128 hi = _mm256_extractf128_ps(t, 1); // a4+a5+a6+a7
    __m128 sum = _mm_add_ps(lo, hi);
    return _mm_cvtss_f32(sum);
}

void MatMulAVX(float matRes[M][K], float matA[M][N], float matB[N][K])
{
    // matB transpose
    float matTmp[K][N];

    __m256 ra0, ra1, ra2, ra3, ra4, ra5, ra6, ra7, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7;

    // transpose matrix B
    MatrixTransposeAVX(matTmp, matB);

    constexpr int mLimit = M - (M % 8);
    constexpr int nLimit = N - (N % 8);
    constexpr int kLimit = K - (K % 8);

    for (int m = 0; m < mLimit; m += 8)
    {
        for (int k = 0; k < kLimit; k += 8)
        {
            __m256 ma00 = _mm256_setzero_ps();
            __m256 ma01 = _mm256_setzero_ps();
            __m256 ma02 = _mm256_setzero_ps();
            __m256 ma03 = _mm256_setzero_ps();
            __m256 ma04 = _mm256_setzero_ps();
            __m256 ma05 = _mm256_setzero_ps();
            __m256 ma06 = _mm256_setzero_ps();
            __m256 ma07 = _mm256_setzero_ps();

            __m256 ma10 = _mm256_setzero_ps();
            __m256 ma11 = _mm256_setzero_ps();
            __m256 ma12 = _mm256_setzero_ps();
            __m256 ma13 = _mm256_setzero_ps();
            __m256 ma14 = _mm256_setzero_ps();
            __m256 ma15 = _mm256_setzero_ps();
            __m256 ma16 = _mm256_setzero_ps();
            __m256 ma17 = _mm256_setzero_ps();

            __m256 ma20 = _mm256_setzero_ps();
            __m256 ma21 = _mm256_setzero_ps();
            __m256 ma22 = _mm256_setzero_ps();
            __m256 ma23 = _mm256_setzero_ps();
            __m256 ma24 = _mm256_setzero_ps();
            __m256 ma25 = _mm256_setzero_ps();
            __m256 ma26 = _mm256_setzero_ps();
            __m256 ma27 = _mm256_setzero_ps();

            __m256 ma30 = _mm256_setzero_ps();
            __m256 ma31 = _mm256_setzero_ps();
            __m256 ma32 = _mm256_setzero_ps();
            __m256 ma33 = _mm256_setzero_ps();
            __m256 ma34 = _mm256_setzero_ps();
            __m256 ma35 = _mm256_setzero_ps();
            __m256 ma36 = _mm256_setzero_ps();
            __m256 ma37 = _mm256_setzero_ps();

            __m256 ma40 = _mm256_setzero_ps();
            __m256 ma41 = _mm256_setzero_ps();
            __m256 ma42 = _mm256_setzero_ps();
            __m256 ma43 = _mm256_setzero_ps();
            __m256 ma44 = _mm256_setzero_ps();
            __m256 ma45 = _mm256_setzero_ps();
            __m256 ma46 = _mm256_setzero_ps();
            __m256 ma47 = _mm256_setzero_ps();

            __m256 ma50 = _mm256_setzero_ps();
            __m256 ma51 = _mm256_setzero_ps();
            __m256 ma52 = _mm256_setzero_ps();
            __m256 ma53 = _mm256_setzero_ps();
            __m256 ma54 = _mm256_setzero_ps();
            __m256 ma55 = _mm256_setzero_ps();
            __m256 ma56 = _mm256_setzero_ps();
            __m256 ma57 = _mm256_setzero_ps();

            __m256 ma60 = _mm256_setzero_ps();
            __m256 ma61 = _mm256_setzero_ps();
            __m256 ma62 = _mm256_setzero_ps();
            __m256 ma63 = _mm256_setzero_ps();
            __m256 ma64 = _mm256_setzero_ps();
            __m256 ma65 = _mm256_setzero_ps();
            __m256 ma66 = _mm256_setzero_ps();
            __m256 ma67 = _mm256_setzero_ps();

            __m256 ma70 = _mm256_setzero_ps();
            __m256 ma71 = _mm256_setzero_ps();
            __m256 ma72 = _mm256_setzero_ps();
            __m256 ma73 = _mm256_setzero_ps();
            __m256 ma74 = _mm256_setzero_ps();
            __m256 ma75 = _mm256_setzero_ps();
            __m256 ma76 = _mm256_setzero_ps();
            __m256 ma77 = _mm256_setzero_ps();

            for (int n = 0; n < nLimit; n += 8)
            {
                // load matrix A
                ra0 = _mm256_loadu_ps(&matA[m][n]);
                ra1 = _mm256_loadu_ps(&matA[m + 1][n]);
                ra2 = _mm256_loadu_ps(&matA[m + 2][n]);
                ra3 = _mm256_loadu_ps(&matA[m + 3][n]);
                ra4 = _mm256_loadu_ps(&matA[m + 4][n]);
                ra5 = _mm256_loadu_ps(&matA[m + 5][n]);
                ra6 = _mm256_loadu_ps(&matA[m + 6][n]);
                ra7 = _mm256_loadu_ps(&matA[m + 7][n]);

                // load matrix B
                rb0 = _mm256_loadu_ps(&matTmp[k][n]);
                rb1 = _mm256_loadu_ps(&matTmp[k + 1][n]);
                rb2 = _mm256_loadu_ps(&matTmp[k + 2][n]);
                rb3 = _mm256_loadu_ps(&matTmp[k + 3][n]);
                rb4 = _mm256_loadu_ps(&matTmp[k + 4][n]);
                rb5 = _mm256_loadu_ps(&matTmp[k + 5][n]);
                rb6 = _mm256_loadu_ps(&matTmp[k + 6][n]);
                rb7 = _mm256_loadu_ps(&matTmp[k + 7][n]);

                // multiply
                ma00 = _mm256_fmadd_ps(ra0, rb0, ma00);
                ma01 = _mm256_fmadd_ps(ra0, rb1, ma01);
                ma02 = _mm256_fmadd_ps(ra0, rb2, ma02);
                ma03 = _mm256_fmadd_ps(ra0, rb3, ma03);
                ma04 = _mm256_fmadd_ps(ra0, rb4, ma04);
                ma05 = _mm256_fmadd_ps(ra0, rb5, ma05);
                ma06 = _mm256_fmadd_ps(ra0, rb6, ma06);
                ma07 = _mm256_fmadd_ps(ra0, rb7, ma07);

                ma10 = _mm256_fmadd_ps(ra1, rb0, ma10);
                ma11 = _mm256_fmadd_ps(ra1, rb1, ma11);
                ma12 = _mm256_fmadd_ps(ra1, rb2, ma12);
                ma13 = _mm256_fmadd_ps(ra1, rb3, ma13);
                ma14 = _mm256_fmadd_ps(ra1, rb4, ma14);
                ma15 = _mm256_fmadd_ps(ra1, rb5, ma15);
                ma16 = _mm256_fmadd_ps(ra1, rb6, ma16);
                ma17 = _mm256_fmadd_ps(ra1, rb7, ma17);

                ma20 = _mm256_fmadd_ps(ra2, rb0, ma20);
                ma21 = _mm256_fmadd_ps(ra2, rb1, ma21);
                ma22 = _mm256_fmadd_ps(ra2, rb2, ma22);
                ma23 = _mm256_fmadd_ps(ra2, rb3, ma23);
                ma24 = _mm256_fmadd_ps(ra2, rb4, ma24);
                ma25 = _mm256_fmadd_ps(ra2, rb5, ma25);
                ma26 = _mm256_fmadd_ps(ra2, rb6, ma26);
                ma27 = _mm256_fmadd_ps(ra2, rb7, ma27);

                ma30 = _mm256_fmadd_ps(ra3, rb0, ma30);
                ma31 = _mm256_fmadd_ps(ra3, rb1, ma31);
                ma32 = _mm256_fmadd_ps(ra3, rb2, ma32);
                ma33 = _mm256_fmadd_ps(ra3, rb3, ma33);
                ma34 = _mm256_fmadd_ps(ra3, rb4, ma34);
                ma35 = _mm256_fmadd_ps(ra3, rb5, ma35);
                ma36 = _mm256_fmadd_ps(ra3, rb6, ma36);
                ma37 = _mm256_fmadd_ps(ra3, rb7, ma37);

                ma40 = _mm256_fmadd_ps(ra4, rb0, ma40);
                ma41 = _mm256_fmadd_ps(ra4, rb1, ma41);
                ma42 = _mm256_fmadd_ps(ra4, rb2, ma42);
                ma43 = _mm256_fmadd_ps(ra4, rb3, ma43);
                ma44 = _mm256_fmadd_ps(ra4, rb4, ma44);
                ma45 = _mm256_fmadd_ps(ra4, rb5, ma45);
                ma46 = _mm256_fmadd_ps(ra4, rb6, ma46);
                ma47 = _mm256_fmadd_ps(ra4, rb7, ma47);

                ma50 = _mm256_fmadd_ps(ra5, rb0, ma50);
                ma51 = _mm256_fmadd_ps(ra5, rb1, ma51);
                ma52 = _mm256_fmadd_ps(ra5, rb2, ma52);
                ma53 = _mm256_fmadd_ps(ra5, rb3, ma53);
                ma54 = _mm256_fmadd_ps(ra5, rb4, ma54);
                ma55 = _mm256_fmadd_ps(ra5, rb5, ma55);
                ma56 = _mm256_fmadd_ps(ra5, rb6, ma56);
                ma57 = _mm256_fmadd_ps(ra5, rb7, ma57);

                ma60 = _mm256_fmadd_ps(ra6, rb0, ma60);
                ma61 = _mm256_fmadd_ps(ra6, rb1, ma61);
                ma62 = _mm256_fmadd_ps(ra6, rb2, ma62);
                ma63 = _mm256_fmadd_ps(ra6, rb3, ma63);
                ma64 = _mm256_fmadd_ps(ra6, rb4, ma64);
                ma65 = _mm256_fmadd_ps(ra6, rb5, ma65);
                ma66 = _mm256_fmadd_ps(ra6, rb6, ma66);
                ma67 = _mm256_fmadd_ps(ra6, rb7, ma67);

                ma70 = _mm256_fmadd_ps(ra7, rb0, ma70);
                ma71 = _mm256_fmadd_ps(ra7, rb1, ma71);
                ma72 = _mm256_fmadd_ps(ra7, rb2, ma72);
                ma73 = _mm256_fmadd_ps(ra7, rb3, ma73);
                ma74 = _mm256_fmadd_ps(ra7, rb4, ma74);
                ma75 = _mm256_fmadd_ps(ra7, rb5, ma75);
                ma76 = _mm256_fmadd_ps(ra7, rb6, ma76);
                ma77 = _mm256_fmadd_ps(ra7, rb7, ma77);
            }
            // horizontal add
            __m256 row0 = _mm256_setr_ps(
                hsum(ma00), hsum(ma01), hsum(ma02), hsum(ma03),
                hsum(ma04), hsum(ma05), hsum(ma06), hsum(ma07));

            __m256 row1 = _mm256_setr_ps(
                hsum(ma10), hsum(ma11), hsum(ma12), hsum(ma13),
                hsum(ma14), hsum(ma15), hsum(ma16), hsum(ma17));

            __m256 row2 = _mm256_setr_ps(
                hsum(ma20), hsum(ma21), hsum(ma22), hsum(ma23),
                hsum(ma24), hsum(ma25), hsum(ma26), hsum(ma27));

            __m256 row3 = _mm256_setr_ps(
                hsum(ma30), hsum(ma31), hsum(ma32), hsum(ma33),
                hsum(ma34), hsum(ma35), hsum(ma36), hsum(ma37));

            __m256 row4 = _mm256_setr_ps(
                hsum(ma40), hsum(ma41), hsum(ma42), hsum(ma43),
                hsum(ma44), hsum(ma45), hsum(ma46), hsum(ma47));

            __m256 row5 = _mm256_setr_ps(
                hsum(ma50), hsum(ma51), hsum(ma52), hsum(ma53),
                hsum(ma54), hsum(ma55), hsum(ma56), hsum(ma57));

            __m256 row6 = _mm256_setr_ps(
                hsum(ma60), hsum(ma61), hsum(ma62), hsum(ma63),
                hsum(ma64), hsum(ma65), hsum(ma66), hsum(ma67));

            __m256 row7 = _mm256_setr_ps(
                hsum(ma70), hsum(ma71), hsum(ma72), hsum(ma73),
                hsum(ma74), hsum(ma75), hsum(ma76), hsum(ma77));

            _mm256_storeu_ps(&matRes[m + 0][k], row0);
            _mm256_storeu_ps(&matRes[m + 1][k], row1);
            _mm256_storeu_ps(&matRes[m + 2][k], row2);
            _mm256_storeu_ps(&matRes[m + 3][k], row3);
            _mm256_storeu_ps(&matRes[m + 4][k], row4);
            _mm256_storeu_ps(&matRes[m + 5][k], row5);
            _mm256_storeu_ps(&matRes[m + 6][k], row6);
            _mm256_storeu_ps(&matRes[m + 7][k], row7);
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

void MatMulAVXPadded(float matRes[paddedM][paddedK], float matA[paddedM][paddedN], float matB[paddedN][paddedK])
{
    // matB transpose
    float matTmp[paddedK][paddedN];

    __m256 ra0, ra1, ra2, ra3, ra4, ra5, ra6, ra7, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7;

    // transpose matrix B
    MatrixTransposeAVXPadded(matTmp, matB);

    for (int m = 0; m < paddedM; m += 8)
    {
        for (int k = 0; k < paddedK; k += 8)
        {
            __m256 ma00 = _mm256_setzero_ps();
            __m256 ma01 = _mm256_setzero_ps();
            __m256 ma02 = _mm256_setzero_ps();
            __m256 ma03 = _mm256_setzero_ps();
            __m256 ma04 = _mm256_setzero_ps();
            __m256 ma05 = _mm256_setzero_ps();
            __m256 ma06 = _mm256_setzero_ps();
            __m256 ma07 = _mm256_setzero_ps();

            __m256 ma10 = _mm256_setzero_ps();
            __m256 ma11 = _mm256_setzero_ps();
            __m256 ma12 = _mm256_setzero_ps();
            __m256 ma13 = _mm256_setzero_ps();
            __m256 ma14 = _mm256_setzero_ps();
            __m256 ma15 = _mm256_setzero_ps();
            __m256 ma16 = _mm256_setzero_ps();
            __m256 ma17 = _mm256_setzero_ps();

            __m256 ma20 = _mm256_setzero_ps();
            __m256 ma21 = _mm256_setzero_ps();
            __m256 ma22 = _mm256_setzero_ps();
            __m256 ma23 = _mm256_setzero_ps();
            __m256 ma24 = _mm256_setzero_ps();
            __m256 ma25 = _mm256_setzero_ps();
            __m256 ma26 = _mm256_setzero_ps();
            __m256 ma27 = _mm256_setzero_ps();

            __m256 ma30 = _mm256_setzero_ps();
            __m256 ma31 = _mm256_setzero_ps();
            __m256 ma32 = _mm256_setzero_ps();
            __m256 ma33 = _mm256_setzero_ps();
            __m256 ma34 = _mm256_setzero_ps();
            __m256 ma35 = _mm256_setzero_ps();
            __m256 ma36 = _mm256_setzero_ps();
            __m256 ma37 = _mm256_setzero_ps();

            __m256 ma40 = _mm256_setzero_ps();
            __m256 ma41 = _mm256_setzero_ps();
            __m256 ma42 = _mm256_setzero_ps();
            __m256 ma43 = _mm256_setzero_ps();
            __m256 ma44 = _mm256_setzero_ps();
            __m256 ma45 = _mm256_setzero_ps();
            __m256 ma46 = _mm256_setzero_ps();
            __m256 ma47 = _mm256_setzero_ps();

            __m256 ma50 = _mm256_setzero_ps();
            __m256 ma51 = _mm256_setzero_ps();
            __m256 ma52 = _mm256_setzero_ps();
            __m256 ma53 = _mm256_setzero_ps();
            __m256 ma54 = _mm256_setzero_ps();
            __m256 ma55 = _mm256_setzero_ps();
            __m256 ma56 = _mm256_setzero_ps();
            __m256 ma57 = _mm256_setzero_ps();

            __m256 ma60 = _mm256_setzero_ps();
            __m256 ma61 = _mm256_setzero_ps();
            __m256 ma62 = _mm256_setzero_ps();
            __m256 ma63 = _mm256_setzero_ps();
            __m256 ma64 = _mm256_setzero_ps();
            __m256 ma65 = _mm256_setzero_ps();
            __m256 ma66 = _mm256_setzero_ps();
            __m256 ma67 = _mm256_setzero_ps();

            __m256 ma70 = _mm256_setzero_ps();
            __m256 ma71 = _mm256_setzero_ps();
            __m256 ma72 = _mm256_setzero_ps();
            __m256 ma73 = _mm256_setzero_ps();
            __m256 ma74 = _mm256_setzero_ps();
            __m256 ma75 = _mm256_setzero_ps();
            __m256 ma76 = _mm256_setzero_ps();
            __m256 ma77 = _mm256_setzero_ps();

            for (int n = 0; n < paddedN; n += 8)
            {
                // load matrix A
                ra0 = _mm256_loadu_ps(&matA[m][n]);
                ra1 = _mm256_loadu_ps(&matA[m + 1][n]);
                ra2 = _mm256_loadu_ps(&matA[m + 2][n]);
                ra3 = _mm256_loadu_ps(&matA[m + 3][n]);
                ra4 = _mm256_loadu_ps(&matA[m + 4][n]);
                ra5 = _mm256_loadu_ps(&matA[m + 5][n]);
                ra6 = _mm256_loadu_ps(&matA[m + 6][n]);
                ra7 = _mm256_loadu_ps(&matA[m + 7][n]);

                // load matrix B
                rb0 = _mm256_loadu_ps(&matTmp[k][n]);
                rb1 = _mm256_loadu_ps(&matTmp[k + 1][n]);
                rb2 = _mm256_loadu_ps(&matTmp[k + 2][n]);
                rb3 = _mm256_loadu_ps(&matTmp[k + 3][n]);
                rb4 = _mm256_loadu_ps(&matTmp[k + 4][n]);
                rb5 = _mm256_loadu_ps(&matTmp[k + 5][n]);
                rb6 = _mm256_loadu_ps(&matTmp[k + 6][n]);
                rb7 = _mm256_loadu_ps(&matTmp[k + 7][n]);

                // multiply
                ma00 = _mm256_fmadd_ps(ra0, rb0, ma00);
                ma01 = _mm256_fmadd_ps(ra0, rb1, ma01);
                ma02 = _mm256_fmadd_ps(ra0, rb2, ma02);
                ma03 = _mm256_fmadd_ps(ra0, rb3, ma03);
                ma04 = _mm256_fmadd_ps(ra0, rb4, ma04);
                ma05 = _mm256_fmadd_ps(ra0, rb5, ma05);
                ma06 = _mm256_fmadd_ps(ra0, rb6, ma06);
                ma07 = _mm256_fmadd_ps(ra0, rb7, ma07);

                ma10 = _mm256_fmadd_ps(ra1, rb0, ma10);
                ma11 = _mm256_fmadd_ps(ra1, rb1, ma11);
                ma12 = _mm256_fmadd_ps(ra1, rb2, ma12);
                ma13 = _mm256_fmadd_ps(ra1, rb3, ma13);
                ma14 = _mm256_fmadd_ps(ra1, rb4, ma14);
                ma15 = _mm256_fmadd_ps(ra1, rb5, ma15);
                ma16 = _mm256_fmadd_ps(ra1, rb6, ma16);
                ma17 = _mm256_fmadd_ps(ra1, rb7, ma17);

                ma20 = _mm256_fmadd_ps(ra2, rb0, ma20);
                ma21 = _mm256_fmadd_ps(ra2, rb1, ma21);
                ma22 = _mm256_fmadd_ps(ra2, rb2, ma22);
                ma23 = _mm256_fmadd_ps(ra2, rb3, ma23);
                ma24 = _mm256_fmadd_ps(ra2, rb4, ma24);
                ma25 = _mm256_fmadd_ps(ra2, rb5, ma25);
                ma26 = _mm256_fmadd_ps(ra2, rb6, ma26);
                ma27 = _mm256_fmadd_ps(ra2, rb7, ma27);

                ma30 = _mm256_fmadd_ps(ra3, rb0, ma30);
                ma31 = _mm256_fmadd_ps(ra3, rb1, ma31);
                ma32 = _mm256_fmadd_ps(ra3, rb2, ma32);
                ma33 = _mm256_fmadd_ps(ra3, rb3, ma33);
                ma34 = _mm256_fmadd_ps(ra3, rb4, ma34);
                ma35 = _mm256_fmadd_ps(ra3, rb5, ma35);
                ma36 = _mm256_fmadd_ps(ra3, rb6, ma36);
                ma37 = _mm256_fmadd_ps(ra3, rb7, ma37);

                ma40 = _mm256_fmadd_ps(ra4, rb0, ma40);
                ma41 = _mm256_fmadd_ps(ra4, rb1, ma41);
                ma42 = _mm256_fmadd_ps(ra4, rb2, ma42);
                ma43 = _mm256_fmadd_ps(ra4, rb3, ma43);
                ma44 = _mm256_fmadd_ps(ra4, rb4, ma44);
                ma45 = _mm256_fmadd_ps(ra4, rb5, ma45);
                ma46 = _mm256_fmadd_ps(ra4, rb6, ma46);
                ma47 = _mm256_fmadd_ps(ra4, rb7, ma47);

                ma50 = _mm256_fmadd_ps(ra5, rb0, ma50);
                ma51 = _mm256_fmadd_ps(ra5, rb1, ma51);
                ma52 = _mm256_fmadd_ps(ra5, rb2, ma52);
                ma53 = _mm256_fmadd_ps(ra5, rb3, ma53);
                ma54 = _mm256_fmadd_ps(ra5, rb4, ma54);
                ma55 = _mm256_fmadd_ps(ra5, rb5, ma55);
                ma56 = _mm256_fmadd_ps(ra5, rb6, ma56);
                ma57 = _mm256_fmadd_ps(ra5, rb7, ma57);

                ma60 = _mm256_fmadd_ps(ra6, rb0, ma60);
                ma61 = _mm256_fmadd_ps(ra6, rb1, ma61);
                ma62 = _mm256_fmadd_ps(ra6, rb2, ma62);
                ma63 = _mm256_fmadd_ps(ra6, rb3, ma63);
                ma64 = _mm256_fmadd_ps(ra6, rb4, ma64);
                ma65 = _mm256_fmadd_ps(ra6, rb5, ma65);
                ma66 = _mm256_fmadd_ps(ra6, rb6, ma66);
                ma67 = _mm256_fmadd_ps(ra6, rb7, ma67);

                ma70 = _mm256_fmadd_ps(ra7, rb0, ma70);
                ma71 = _mm256_fmadd_ps(ra7, rb1, ma71);
                ma72 = _mm256_fmadd_ps(ra7, rb2, ma72);
                ma73 = _mm256_fmadd_ps(ra7, rb3, ma73);
                ma74 = _mm256_fmadd_ps(ra7, rb4, ma74);
                ma75 = _mm256_fmadd_ps(ra7, rb5, ma75);
                ma76 = _mm256_fmadd_ps(ra7, rb6, ma76);
                ma77 = _mm256_fmadd_ps(ra7, rb7, ma77);
            }
            // horizontal add
            __m256 row0 = _mm256_setr_ps(
                hsum(ma00), hsum(ma01), hsum(ma02), hsum(ma03),
                hsum(ma04), hsum(ma05), hsum(ma06), hsum(ma07));

            __m256 row1 = _mm256_setr_ps(
                hsum(ma10), hsum(ma11), hsum(ma12), hsum(ma13),
                hsum(ma14), hsum(ma15), hsum(ma16), hsum(ma17));

            __m256 row2 = _mm256_setr_ps(
                hsum(ma20), hsum(ma21), hsum(ma22), hsum(ma23),
                hsum(ma24), hsum(ma25), hsum(ma26), hsum(ma27));

            __m256 row3 = _mm256_setr_ps(
                hsum(ma30), hsum(ma31), hsum(ma32), hsum(ma33),
                hsum(ma34), hsum(ma35), hsum(ma36), hsum(ma37));

            __m256 row4 = _mm256_setr_ps(
                hsum(ma40), hsum(ma41), hsum(ma42), hsum(ma43),
                hsum(ma44), hsum(ma45), hsum(ma46), hsum(ma47));

            __m256 row5 = _mm256_setr_ps(
                hsum(ma50), hsum(ma51), hsum(ma52), hsum(ma53),
                hsum(ma54), hsum(ma55), hsum(ma56), hsum(ma57));

            __m256 row6 = _mm256_setr_ps(
                hsum(ma60), hsum(ma61), hsum(ma62), hsum(ma63),
                hsum(ma64), hsum(ma65), hsum(ma66), hsum(ma67));

            __m256 row7 = _mm256_setr_ps(
                hsum(ma70), hsum(ma71), hsum(ma72), hsum(ma73),
                hsum(ma74), hsum(ma75), hsum(ma76), hsum(ma77));

            _mm256_storeu_ps(&matRes[m + 0][k], row0);
            _mm256_storeu_ps(&matRes[m + 1][k], row1);
            _mm256_storeu_ps(&matRes[m + 2][k], row2);
            _mm256_storeu_ps(&matRes[m + 3][k], row3);
            _mm256_storeu_ps(&matRes[m + 4][k], row4);
            _mm256_storeu_ps(&matRes[m + 5][k], row5);
            _mm256_storeu_ps(&matRes[m + 6][k], row6);
            _mm256_storeu_ps(&matRes[m + 7][k], row7);
        }
    }
}

bool verify(float matScalar[M][K], float matVec[M][K])
{
    constexpr float absTolerance = 1e-3f;
    constexpr float relTolerance = 2e-6f;

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
    memset(matrixResV, 0, sizeof(matrixResV));

    // cout << "\nResultant Matrix: Cleaned" << endl;

    cout << "Running AVX code" << endl;
    // AVX version call
    // // warmup run
    for (int i = 0; i < warmup_iter; i++)
    {
        MatMulAVX(matrixResV, matrixA, matrixB);
    }

    // actual run
    auto avx_start = chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++)
    {
        MatMulAVX(matrixResV, matrixA, matrixB);
    }
    // MatrixTranspose(matrixA, matrixB);
    auto avx_end = chrono::high_resolution_clock::now();

    auto avx_duration = chrono::duration_cast<chrono::nanoseconds>(avx_end - avx_start);

    cout << "\nAVX Resultant Matrix:" << endl;
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
    memset(matrixResV, 0, sizeof(matrixResV));

    cout << "Running AVX Padded code" << endl;
    // AVX Padded version call
    // // warmup run
    for (int i = 0; i < warmup_iter; i++)
    {
        MatMulAVXPadded(matResPad, matAPad, matBPad);
    }

    // actual run
    auto avx_pad_start = chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++)
    {
        MatMulAVXPadded(matResPad, matAPad, matBPad);
    }
    auto avx_pad_end = chrono::high_resolution_clock::now();

    auto avx_pad_duration = chrono::duration_cast<chrono::nanoseconds>(avx_pad_end - avx_pad_start);

    cout << "\nAVX Padded Resultant Matrix:" << endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            matrixResV[i][j] = matResPad[i][j];
            cout << matrixResV[i][j] << " ";
        }
        cout << endl;
    }
    if (!verify(matrixRes, matrixResV))
    {
        return -1;
    }

    cout << "\nResult: " << endl;
    cout << "Scalar Version Time: " << scalar_duration.count() / 10 << " ns" << endl;
    cout << "AVX Version Time: " << avx_duration.count() / 10 << " ns" << endl;
    cout << "AVX padded Version Time: " << avx_pad_duration.count() / 10 << " ns" << endl;
}
