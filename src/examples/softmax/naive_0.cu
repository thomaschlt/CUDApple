// naive CUDA kernel for softmax (not working atm)
__global__ void softmax_kernel_0(float *matd, float *resd, int M, int N)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M)
    {
        // max
        float m = -1 * INFINITY;
        // norm factor
        float L = 0.0f;

        // 3 passes (not optimal)
        for (int col = 0; col < N; col++)
        {
            int i = row * N + col;
            m = max(m, matd[i]);
        }
        for (int col = 0; col < N; col++)
        {
            int i = row * N + col;
            L += expf(matd[i] - m);
        }
        for (int col = 0; col < N; col++)
        {
            int i = row * N + col;
            resd[i] = expf(matd[i] - m) / L;
        }
    }
}