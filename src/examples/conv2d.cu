__device__ float relu_activation(float x)
{
    return fmaxf(0.0f, x);
}

// single channel, 3x3 kernel
__global__ void conv2d_simple(
    float *input,
    float *kernel,
    float *output,
    int H, int W)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int H_out = H - 2;
    int W_out = W - 2;

    if (row < H_out && col < W_out)
    {
        float sum = 0.0f;

        for (int kh = 0; kh < 3; kh++)
        {
            for (int kw = 0; kw < 3; kw++)
            {
                int in_row = row + kh;
                int in_col = col + kw;
                int in_idx = in_row * W + in_col;
                int k_idx = kh * 3 + kw;

                sum = sum + input[in_idx] * kernel[k_idx];
            }
        }

        int out_idx = row * W_out + col;
        output[out_idx] = relu_activation(sum);
    }
}

__global__ void conv2d_advanced(
    float *input,
    float *weight,
    float *bias,
    float *output,
    int H, int W, int C,
    int H_out, int W_out,
    int K_h, int K_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w)
{
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_row < H_out && out_col < W_out)
    {
        for (int ch = 0; ch < C; ch++)
        {
            float sum = 0.0f;

            if (bias != 0)
            {
                sum = bias[0];
            }

            for (int kh = 0; kh < K_h; kh++)
            {
                for (int kw = 0; kw < K_w; kw++)
                {
                    int in_row = out_row * stride_h + kh - pad_h;
                    int in_col = out_col * stride_w + kw - pad_w;

                    if (in_row >= 0 && in_row < H && in_col >= 0 && in_col < W)
                    {
                        int in_idx = in_row * (W * C) + in_col * C + ch;
                        int w_idx = kh * (K_w * C) + kw * C + ch;
                        sum = sum + input[in_idx] * weight[w_idx];
                    }
                }
            }

            int out_idx = out_row * (W_out * C) + out_col * C + ch;
            output[out_idx] = relu_activation(sum);
        }
    }
}