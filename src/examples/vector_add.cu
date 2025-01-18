// Kernel for vector addition
__global__ void vectorAdd(float *vec1, float *vec2, float *res, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim)
    {
        res[i] = vec1[i] + vec2[i];
    }
}