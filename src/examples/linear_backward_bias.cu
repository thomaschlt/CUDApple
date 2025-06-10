__global__ void linear_backward_bias(
    float *grad_output, // [batch_size, out_features]
    float *grad_bias,   // [out_features]
    int batch_size,
    int out_features)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx < out_features)
    {
        float sum = 0.0f;

        // Sum across batch dimension
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++)
        {
            sum += grad_output[batch_idx * out_features + out_idx];
        }

        grad_bias[out_idx] = sum;
    }
}