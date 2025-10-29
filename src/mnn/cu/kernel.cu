#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda.h
#include "include/mnn.hpp"

/// ----------------- Forward Propagation ----------------- ///
__global__ void kernelLayerForward1(float* input,
                                    float* weights,
                                    float* biases,
                                    float* output,
                                    int input_size,
                                    int output_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // neuron index in current layer

    if (j < output_size) {
        float sum = biases[j];
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weights[j * input_size + i];
        }
        output[j] = sum;
    }
}

__global__ void kernelLayerForward2(float* input,
                                    float* weights,
                                    float* biases,
                                    float* output,
                                    int input_size,
                                    int output_size,
                                    float power)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // neuron index in current layer

    if (j < output_size) {
        float sum = biases[j];
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weights[j * input_size + i];
        }
        // Example activation function: ReLU
        output[j] = fmaxf(0.0f, sum);
    }
}

__global__ void kernelLayerForward3(float* input,
                                    float* weights,
                                    float* biases,
                                    float* output,
                                    int inHeight,
                                    int inWidth,
                                    int output_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // neuron index in current layer

    if (j < output_size) {
        float sum = biases[j];
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weights[j * input_size + i];
        }
        // Example activation function: Sigmoid
        output[j] = 1.0f / (1.0f + expf(-sum));
    }
}

__global__ void kernelLayerForward4(float* input,
                                    float* weights,
                                    float* biases,
                                    float* output,
                                    int inHeight,
                                    int inWidth,
                                    int output_size,
                                    float power)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // neuron index
    if (j < output_size) {
        float sum = biases[j];
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weights[j * input_size + i];
        }
        // Example activation function: Tanh
        output[j] = tanhf(sum);
    }
}


/// ----------------- Backpropagation ----------------- ///

__device__ __inline__ void kernelGradient(double *c, double *b, double x, double L, int n) {
    double factor = 1.0 - L * (n - 1.0) / x;
    double old_c = *c;
    *c = 0.9 * factor * old_c;
    *b = 0.1 * factor * old_c;
}

/// ----------------- Weight Update ----------------- ///

__global__ void kernelUpdateWeights(float* weights,
                                    float* gweights,
                                    float learning_rate,
                                    int current_layer_size,
                                    int prev_layer_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // neuron index in previous layer ('col')
    int j = blockIdx.y * blockDim.y + threadIdx.y; // neuron index in current layer ('row')

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i;
        weights[weight_idx] -= learning_rate * gweights[weight_idx];
    }
}


__global__ void kernelUpdateWeightsWithL1(float* weights,
                                          float* gweights,
                                          int current_layer_size,
                                          int prev_layer_size,
                                          float learning_rate,
                                          float lambda_l1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // neuron index in previous layer ('col')
    int j = blockIdx.y * blockDim.y + threadIdx.y; // neuron index in current layer ('row')

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i;

        float current_weight = weights[weight_idx];

        // Subgradient of the L1 regularization term
        float sign;
        if (current_weight < 0.0f) {
            sign = -1.0f;
        }
        else if (current_weight > 0.0f) {
            sign = 1.0f;
        }
        else {
            sign = 0.0f;
        }

        float l1_gradient = sign * lambda_l1;

        // Total gradient
        float total_gradient = gweights[weight_idx] + l1_gradient;

        weights[weight_idx] -= learning_rate * total_gradient;
    }
}


__global__ void kernelUpdateWeightsWithL2(float* weights,
                                          float* gweights,
                                          int current_layer_size,
                                          int prev_layer_size,
                                          float learning_rate,
                                          float lambda_l2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // neuron index in previous layer ('col')
    int j = blockIdx.y * blockDim.y + threadIdx.y; // neuron index in current layer ('row')

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i;

        // Gradient of the L2 regularization term
        float current_weight = weights[weight_idx];
        float l2_gradient = lambda_l2 * current_weight;

        // Total gradient
        float total_gradient = gweights[weight_idx] + l2_gradient;

        weights[weight_idx] -= learning_rate * total_gradient;
    }
}


__global__ void kernelUpdateWeightsElasticNet(float* weights,
                                              float* gweights,
                                              int current_layer_size,
                                              int prev_layer_size,
                                              float learning_rate,
                                              float lambda_l1,
                                              float lambda_l2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // neuron index in previous layer ('col')
    int j = blockIdx.y * blockDim.y + threadIdx.y; // neuron index in current layer ('row')

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i;

        // Gradient of the L2 regularization term
        float current_weight = weights[weight_idx];
        float l2_gradient = lambda_l2 * current_weight;

        // Subgradient of the L1 regularization term
        float sign;
        if (current_weight < 0.0f) {
            sign = -1.0f;
        }
        else if (current_weight > 0.0f) {
            sign = 1.0f;
        }
        else {
            sign = 0.0f; // Subgradient is 0 when weight is 0
        }

        float l1_gradient = sign * lambda_l1;

        // Total gradient
        float total_gradient = gweights[weight_idx] + l2_gradient + l1_gradient;

        weights[weight_idx] -= learning_rate * total_gradient;
    }
}


__global__ void kernelUpdateWeightsWithWeightDecay(float* weights,
                                                   float* gweights,
                                                   int current_layer_size,
                                                   int prev_layer_size,
                                                   float learning_rate,
                                                   float decay_rate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // neuron index in previous layer ('col')
    int j = blockIdx.y * blockDim.y + threadIdx.y; // neuron index in current layer ('row')

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i;

        float current_weight = weights[weight_idx];
        float gradient = gweights[weight_idx];

        weights[weight_idx] = current_weight * (1.0f - decay_rate) - learning_rate * gradient;
    }
}


__global__ void kernelUpdateWeightsDropout(float* weights,
                                           float* gweights,
                                           int current_layer_size,
                                           int prev_layer_size,
                                           float learning_rate,
                                           float dropout_rate,
                                           unsigned int base_seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // neuron index in previous layer ('col')
    int j = blockIdx.y * blockDim.y + threadIdx.y; // neuron index in current layer ('row')

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i;

        unsigned int seed = base_seed + weight_idx;
        seed = (seed * 1664525u + 1013904223u);
        float rand_val = (float)(seed) / (float)(0xFFFFFFFF);

        if (rand_val >= dropout_rate) {
            weights[weight_idx] -= learning_rate * gweights[weight_idx];
        }
    }
}

#endif // KERNEL_CU