__global__ void matrixAdd(float *A, float *B, float *C, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}