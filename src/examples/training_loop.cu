__global__ void mnist_training_step(
    // Input data
    float *input_batch,   // [batch_size, 784] - Flattened input images
    float *target_labels, // [batch_size, 10] - One-hot target labels

    // Simple 2-layer network parameters
    float *fc1_weights, // [784, 128] - First layer weights
    float *fc1_bias,    // [128] - First layer bias
    float *fc2_weights, // [128, 10] - Output layer weights
    float *fc2_bias,    // [10] - Output layer bias

    // Intermediate activations
    float *fc1_output,  // [batch_size, 128] - Hidden layer output
    float *fc2_output,  // [batch_size, 10] - Logits
    float *predictions, // [batch_size, 10] - Softmax predictions
    float *loss_output, // [batch_size] - Loss per sample

    // Gradient buffers
    float *grad_fc2_weights, // [128, 10] - Gradients for output weights
    float *grad_fc2_bias,    // [10] - Gradients for output bias
    float *grad_fc1_weights, // [784, 128] - Gradients for hidden weights
    float *grad_fc1_bias,    // [128] - Gradients for hidden bias

    // Training parameters
    float *learning_rate, // [1] - Learning rate
    int batch_size,       // Batch size
    int input_size,       // 784 for MNIST
    int hidden_size,      // 128
    int output_size)      // 10 for MNIST classes
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size)
        return;

    // ===========================
    // FORWARD PASS
    // ===========================

    // 1. First Linear Layer: input → hidden (with ReLU)
    for (int h = 0; h < hidden_size; h++)
    {
        float sum = fc1_bias[h];
        for (int i = 0; i < input_size; i++)
        {
            sum += input_batch[batch_idx * input_size + i] * fc1_weights[i * hidden_size + h];
        }
        fc1_output[batch_idx * hidden_size + h] = fmaxf(0.0f, sum); // ReLU
    }

    // 2. Second Linear Layer: hidden → output
    for (int o = 0; o < output_size; o++)
    {
        float sum = fc2_bias[o];
        for (int h = 0; h < hidden_size; h++)
        {
            sum += fc1_output[batch_idx * hidden_size + h] * fc2_weights[h * output_size + o];
        }
        fc2_output[batch_idx * output_size + o] = sum; // Logits
    }

    // 3. Softmax activation
    float max_logit = fc2_output[batch_idx * output_size];
    for (int o = 1; o < output_size; o++)
    {
        max_logit = fmaxf(max_logit, fc2_output[batch_idx * output_size + o]);
    }

    float sum_exp = 0.0f;
    for (int o = 0; o < output_size; o++)
    {
        predictions[batch_idx * output_size + o] = expf(fc2_output[batch_idx * output_size + o] - max_logit);
        sum_exp += predictions[batch_idx * output_size + o];
    }

    for (int o = 0; o < output_size; o++)
    {
        predictions[batch_idx * output_size + o] /= sum_exp;
    }

    // 4. Cross-entropy loss
    float loss = 0.0f;
    for (int o = 0; o < output_size; o++)
    {
        if (target_labels[batch_idx * output_size + o] > 0.5f)
        {
            loss -= logf(predictions[batch_idx * output_size + o] + 1e-8f);
        }
    }
    loss_output[batch_idx] = loss;

    // ===========================
    // BACKWARD PASS
    // ===========================

    // Gradient of softmax + cross-entropy
    float grad_output[10];
    for (int o = 0; o < output_size; o++)
    {
        grad_output[o] = predictions[batch_idx * output_size + o] - target_labels[batch_idx * output_size + o];
    }

    // Gradients for FC2 (output layer)
    for (int h = 0; h < hidden_size; h++)
    {
        for (int o = 0; o < output_size; o++)
        {
            int weight_idx = h * output_size + o;
            atomicAdd(&grad_fc2_weights[weight_idx],
                      fc1_output[batch_idx * hidden_size + h] * grad_output[o]);
        }
    }

    for (int o = 0; o < output_size; o++)
    {
        atomicAdd(&grad_fc2_bias[o], grad_output[o]);
    }

    // Gradients for FC1 (hidden layer)
    float grad_hidden[128];
    for (int h = 0; h < hidden_size; h++)
    {
        grad_hidden[h] = 0.0f;
        for (int o = 0; o < output_size; o++)
        {
            grad_hidden[h] += fc2_weights[h * output_size + o] * grad_output[o];
        }
        // ReLU derivative
        if (fc1_output[batch_idx * hidden_size + h] <= 0.0f)
        {
            grad_hidden[h] = 0.0f;
        }
    }

    for (int i = 0; i < input_size; i++)
    {
        for (int h = 0; h < hidden_size; h++)
        {
            int weight_idx = i * hidden_size + h;
            atomicAdd(&grad_fc1_weights[weight_idx],
                      input_batch[batch_idx * input_size + i] * grad_hidden[h]);
        }
    }

    for (int h = 0; h < hidden_size; h++)
    {
        atomicAdd(&grad_fc1_bias[h], grad_hidden[h]);
    }

    // ===========================
    // PARAMETER UPDATES (SGD)
    // ===========================

    // Only the first thread in the block updates parameters
    if (batch_idx == 0)
    {
        // Update FC1 weights
        for (int i = 0; i < input_size * hidden_size; i++)
        {
            fc1_weights[i] = fc1_weights[i] - learning_rate[0] * grad_fc1_weights[i];
            grad_fc1_weights[i] = 0.0f; // Reset for next iteration
        }

        // Update FC1 bias
        for (int h = 0; h < hidden_size; h++)
        {
            fc1_bias[h] = fc1_bias[h] - learning_rate[0] * grad_fc1_bias[h];
            grad_fc1_bias[h] = 0.0f;
        }

        // Update FC2 weights
        for (int i = 0; i < hidden_size * output_size; i++)
        {
            fc2_weights[i] = fc2_weights[i] - learning_rate[0] * grad_fc2_weights[i];
            grad_fc2_weights[i] = 0.0f;
        }

        // Update FC2 bias
        for (int o = 0; o < output_size; o++)
        {
            fc2_bias[o] = fc2_bias[o] - learning_rate[0] * grad_fc2_bias[o];
            grad_fc2_bias[o] = 0.0f;
        }
    }
}

// ===========================
// TRAINING COORDINATOR
// ===========================

__global__ void run_training_epoch(
    float *train_data,   // [num_samples, 784] - Training images
    float *train_labels, // [num_samples, 10] - Training labels
    float *fc1_weights,  // Model parameters
    float *fc1_bias,
    float *fc2_weights,
    float *fc2_bias,
    float *epoch_loss,   // [1] - Average loss for this epoch
    float learning_rate, // Learning rate
    int num_samples,     // Total training samples
    int batch_size,      // Batch size
    int epoch_num)       // Current epoch number
{
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample_idx >= num_samples)
        return;

    // For simplicity, process one sample at a time
    // In practice, you'd batch this

    // Allocate temporary buffers (in shared memory for efficiency)
    __shared__ float shared_fc1_output[128];
    __shared__ float shared_fc2_output[10];
    __shared__ float shared_predictions[10];
    __shared__ float shared_gradients[1000]; // For various gradients

    // Process this sample through the network
    // (Implementation would call mnist_training_step)

    // Accumulate loss
    // atomicAdd(epoch_loss, sample_loss / num_samples);
}
