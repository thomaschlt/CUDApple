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