__global__ void linear_backward_weights(
    float *input,        // [batch_size, in_features]
    float *grad_output,  // [batch_size, out_features]
    float *grad_weights, // [in_features, out_features]
    int batch_size,
    int in_features,
    int out_features)
{
    int in_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (in_idx < in_features && out_idx < out_features)
    {
        float sum = 0.0f;

        // Matrix multiplication: input[:, in_idx].T @ grad_output[:, out_idx]
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++)
        {
            sum += input[batch_idx * in_features + in_idx] *
                   grad_output[batch_idx * out_features + out_idx];
        }

        grad_weights[in_idx * out_features + out_idx] = sum;
    }
}