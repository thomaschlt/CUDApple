__device__ float test_basic_math(float x)
{
    return fmaxf(0.0f, x); // ReLU
}

__device__ float test_exponential(float x)
{
    return expf(x); // Exponential
}

__device__ float test_logarithm(float x)
{
    return logf(fmaxf(x, 0.001f)); // Log with safety check
}

__device__ float test_square_root(float x)
{
    return sqrtf(fmaxf(x, 0.0f)); // Sqrt with safety check
}

__device__ float test_power(float x, float power)
{
    return powf(fmaxf(x, 0.0f), power); // Power function
}

__device__ float test_hyperbolic_tangent(float x)
{
    return tanhf(x); // Hyperbolic tangent
}

__device__ float test_absolute_value(float x)
{
    return fabsf(x); // Absolute value
}

__device__ float test_min_max(float x, float y)
{
    return fmaxf(fminf(x, y), 0.0f); // Combined min/max
}

// Complex activation functions
__device__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ float leaky_relu(float x)
{
    return fmaxf(0.01f * x, x);
}

__device__ float gelu(float x)
{
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    float x_cubed = powf(x, 3.0f);
    float inner = x + 0.044715f * x_cubed;
    float scaled = 0.7978845608f * inner; // sqrt(2/π) ≈ 0.7978845608
    return 0.5f * x * (1.0f + tanhf(scaled));
}

__device__ float swish(float x)
{
    return x * sigmoid(x);
}

__global__ void test_relu(float *input, float *output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output[i] = test_basic_math(input[i]);
    }
}

__global__ void test_exp(float *input, float *output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output[i] = test_exponential(input[i] * 0.1f); // Scale to avoid overflow
    }
}

__global__ void test_log(float *input, float *output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output[i] = test_logarithm(fabsf(input[i]) + 1.0f); // Ensure positive
    }
}

__global__ void test_sqrt(float *input, float *output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output[i] = test_square_root(fabsf(input[i]));
    }
}

__global__ void test_pow(float *input, float *output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output[i] = test_power(fabsf(input[i]), 2.0f); // Square function
    }
}

__global__ void test_tanh(float *input, float *output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output[i] = test_hyperbolic_tangent(input[i]);
    }
}

__global__ void test_abs(float *input, float *output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output[i] = test_absolute_value(input[i]);
    }
}

__global__ void test_sigmoid(float *input, float *output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output[i] = sigmoid(input[i]);
    }
}

__global__ void test_gelu(float *input, float *output, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        output[i] = gelu(input[i]);
    }
}