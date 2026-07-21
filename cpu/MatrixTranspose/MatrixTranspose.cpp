#include <iostream>
#include <vector>

#define M 21
#define N 21

using namespace std;

// Scalar version
void MatrixTranspose(float matrixA[][N], float matrixB[][M])
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrixB[j][i] = matrixA[i][j];
        }
    }
}

int main()
{
    float matrixA[M][N];
    float matrixB[N][M];

    cout << "Matrix A: Before Transpose" << endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrixA[i][j] = (N * i) + j;
            cout << " " << matrixA[i][j] << " ";
        }
        cout << endl;
    }

    // scalar call
    MatrixTranspose(matrixA, matrixB);

    cout << "Matrix B: After Transpose" << endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            cout << " " << matrixB[i][j] << " ";
        }
        cout << endl;
    }
}