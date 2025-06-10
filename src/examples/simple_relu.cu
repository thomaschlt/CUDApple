__device__ float relu(float x)
{
    return fmaxf(0.0f, x);
}

__global__ void apply_relu(float *input, float *output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output[i] = relu(input[i]);
    }
}