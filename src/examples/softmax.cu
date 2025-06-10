__global__ void softmax_forward(
    float *input,
    float *output,
    int batch_size,
    int num_classes)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size)
    {
        int offset = batch_idx * num_classes;

        float max_val = input[offset];
        for (int i = 1; i < num_classes; i++)
        {
            max_val = fmaxf(max_val, input[offset + i]);
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++)
        {
            sum_exp += expf(input[offset + i] - max_val);
        }

        for (int i = 0; i < num_classes; i++)
        {
            output[offset + i] = expf(input[offset + i] - max_val) / sum_exp;
        }
    }
}

__global__ void cross_entropy_loss(
    float *predictions,
    float *targets,
    float *loss,
    int batch_size,
    int num_classes)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size)
    {
        int offset = batch_idx * num_classes;
        float sample_loss = 0.0f;

        for (int i = 0; i < num_classes; i++)
        {
            if (targets[offset + i] > 0.0f)
            {
                // small epsilon to prevent log(0)
                float pred = fmaxf(predictions[offset + i], 0.0000001f);
                float log_pred = logf(pred);
                float target_val = targets[offset + i];
                float loss_term = target_val * log_pred;
                sample_loss = sample_loss - loss_term;
            }
        }

        loss[batch_idx] = sample_loss;
    }
}
