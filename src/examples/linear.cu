__global__ void linear_forward(
    float *input,
    float *weight,
    float *bias,
    float *output,
    int batch_size,
    int in_features,
    int out_features)
{
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && out_idx < out_features)
    {
        float sum = 0.0f;

        // mat mul: output = input @ weight.T + bias
        for (int in_idx = 0; in_idx < in_features; in_idx++)
        {
            int input_offset = batch_idx * in_features + in_idx;
            int weight_offset = out_idx * in_features + in_idx;
            sum += input[input_offset] * weight[weight_offset];
        }

        if (bias != NULL)
        {
            sum += bias[out_idx];
        }

        int output_offset = batch_idx * out_features + out_idx;
        output[output_offset] = sum;
    }
}