__global__ void conv2d_backward_weights(
    float *input,        // [H, W, C_in] - Input feature maps
    float *grad_output,  // [H_out, W_out, C_out] - Gradients from next layer
    float *grad_weights, // [K_h, K_w, C_in, C_out] - Output: weight gradients
    int H, int W, int C_in,
    int H_out, int W_out, int C_out,
    int K_h, int K_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w)
{
    // Each thread computes gradient for one weight element
    int kh = blockIdx.y * blockDim.y + threadIdx.y;
    int kw = blockIdx.x * blockDim.x + threadIdx.x;

    if (kh < K_h && kw < K_w)
    {
        for (int c_in = 0; c_in < C_in; c_in++)
        {
            for (int c_out = 0; c_out < C_out; c_out++)
            {
                float weight_gradient = 0.0f;

                // Iterate over all output positions
                for (int h_out = 0; h_out < H_out; h_out++)
                {
                    for (int w_out = 0; w_out < W_out; w_out++)
                    {
                        // Calculate corresponding input position
                        int h_in = h_out * stride_h - pad_h + kh;
                        int w_in = w_out * stride_w - pad_w + kw;

                        // Check if input position is valid
                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
                        {
                            weight_gradient +=
                                input[h_in * W * C_in + w_in * C_in + c_in] *
                                grad_output[h_out * W_out * C_out + w_out * C_out + c_out];
                        }
                    }
                }

                grad_weights[kh * K_w * C_in * C_out + kw * C_in * C_out + c_in * C_out + c_out] = weight_gradient;
            }
        }
    }
}