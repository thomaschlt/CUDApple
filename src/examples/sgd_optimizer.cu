__global__ void sgd_optimizer(
    float *weights,      // [N] - Model weights (in/out)
    float *grad_weights, // [N] - Weight gradients (input)
    float *lr,           // [1] - Learning rate as single-element array
    int N)               // Size of weight array
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        weights[idx] = weights[idx] - lr[0] * grad_weights[idx];
    }
}