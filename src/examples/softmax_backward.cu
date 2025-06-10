__global__ void softmax_cross_entropy_backward(
    float *grad_output,
    float *predictions,
    float *targets,
    float *grad_input,
    int batch_size,
    int num_classes)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size)
    {
        int offset = batch_idx * num_classes;
        float batch_grad_scale = grad_output[batch_idx];

        for (int i = 0; i < num_classes; i++)
        {
            // grad = (prediction - target) * grad_from_above
            grad_input[offset + i] = (predictions[offset + i] - targets[offset + i]) * batch_grad_scale;
        }
    }
}