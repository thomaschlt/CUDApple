__global__ void conv2d_backward_bias(
    float *grad_output, // [H_out, W_out, C_out] - Gradients from next layer
    float *grad_bias,   // [C_out] - Output: bias gradients
    int H_out, int W_out, int C_out)
{
    // Each thread computes gradient for one bias element (one output channel)
    int c_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (c_out < C_out)
    {
        float bias_gradient = 0.0f;

        // Sum gradients across all spatial positions for this channel
        for (int h = 0; h < H_out; h++)
        {
            for (int w = 0; w < W_out; w++)
            {
                bias_gradient += grad_output[h * W_out * C_out + w * C_out + c_out];
            }
        }

        grad_bias[c_out] = bias_gradient;
    }
}
