__global__ void maxpool2d_forward(
    float *input,
    float *output,
    int H, int W, int C,
    int H_out, int W_out,
    int pool_h, int pool_w,
    int stride_h, int stride_w)
{
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_row < H_out && out_col < W_out)
    {
        for (int ch = 0; ch < C; ch++)
        {
            float max_val = -INFINITY;

            for (int ph = 0; ph < pool_h; ph++)
            {
                for (int pw = 0; pw < pool_w; pw++)
                {
                    int in_row = out_row * stride_h + ph;
                    int in_col = out_col * stride_w + pw;

                    if (in_row < H && in_col < W)
                    {
                        int in_idx = in_row * (W * C) + in_col * C + ch;
                        max_val = fmaxf(max_val, input[in_idx]);
                    }
                }
            }

            int out_idx = out_row * (W_out * C) + out_col * C + ch;
            output[out_idx] = max_val;
        }
    }
}