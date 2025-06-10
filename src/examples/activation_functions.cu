__device__ float relu(float x)
{
    return fmaxf(0.0f, x);
}

__device__ float tanh_activation(float x)
{
    return tanhf(x);
}

__device__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ float leaky_relu(float x)
{
    return fmaxf(0.01f * x, x);
}

__global__ void apply_relu(float *input, float *output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output[i] = relu(input[i]);
    }
}

__global__ void apply_tanh(float *input, float *output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output[i] = tanh_activation(input[i]);
    }
}