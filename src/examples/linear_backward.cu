__global__ void linear_backward_input(
    float *grad_output, // [batch_size, out_features]
    float *weights,     // [in_features, out_features]
    float *grad_input,  // [batch_size, in_features]
    int batch_size,
    int in_features,
    int out_features)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int in_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < batch_size && in_idx < in_features)
    {
        float sum = 0.0f;

        // Matrix multiplication: grad_output[batch_idx, :] @ weights[in_idx, :]
        for (int out_idx = 0; out_idx < out_features; out_idx++)
        {
            sum += grad_output[batch_idx * out_features + out_idx] *
                   weights[in_idx * out_features + out_idx];
        }

        grad_input[batch_idx * in_features + in_idx] = sum;
    }
}
