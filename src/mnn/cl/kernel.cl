__kernel void kernelUpdateWeights(__global float* weights, 
                                __global float* gweights,
                                float learning_rate,
                                int current_layer_size,
                                int prev_layer_size)
{
    int j = get_global_id(0);
    int i = get_global_id(1);

    if (j < current_layer_size && i < prev_layer_size) {        
        gweights[weight_idx] = gradient;
        weights[weight_idx] -= learning_rate * gweights[weight_idx];
    }
}


__kernel void kernelUpdateWeightsWithL1(__global float* weights,
                                        __global float* gweights, 
                                        int current_layer_size, 
                                        int prev_layer_size,
                                        float learning_rate, 
                                        float lambda_l1)
{
    // Consistent 2D indexing with other update kernels
    int i = get_global_id(0); // neuron index in previous layer ('col')
    int j = get_global_id(1); // neuron index in current layer ('row')

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


__kernel void kernelUpdateWeightsWithL2(__global float* weights,
                                        __global float* gweights, 
                                        int current_layer_size, 
                                        int prev_layer_size,
                                        float learning_rate, 
                                        float lambda_l2)
{
    // Consistent 2D indexing with other update kernels
    int i = get_global_id(0); // neuron index in previous layer ('col')
    int j = get_global_id(1); // neuron index in current layer ('row')

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


__kernel void kernelUpdateWeightsElasticNet(__global float* weights,
                                            __global float* gweights, 
                                            int current_layer_size, 
                                            int prev_layer_size,
                                            float learning_rate, 
                                            float lambda_l1,
                                            float lambda_l2)
{
    // Consistent 2D indexing with other update kernels
    int i = get_global_id(0); // neuron index in previous layer ('col')
    int j = get_global_id(1); // neuron index in current layer ('row')

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


__kernel void kernelUpdateWeightsWithWeightDecay(__global float* weights,
                                                 __global float* gweights,
                                                 int current_layer_size,
                                                 int prev_layer_size,
                                                 float learning_rate,
                                                 float decay_rate)
{
    // Consistent 2D indexing with other update kernels
    int i = get_global_id(0); // neuron index in previous layer ('col')
    int j = get_global_id(1); // neuron index in current layer ('row')

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i;

        float current_weight = weights[weight_idx];
        float gradient = gweights[weight_idx];

        weights[weight_idx] = current_weight * (1.0f - decay_rate) - learning_rate * gradient;
    }
}


__kernel void kernelUpdateWeightsDropout(__global float* weights,
                                         __global float* gweights,
                                         int current_layer_size,
                                         int prev_layer_size,
                                         float learning_rate,
                                         float dropout_rate,
                                         uint base_seed)
{
    int i = get_global_id(0); // neuron index in previous layer ('col')
    int j = get_global_id(1); // neuron index in current layer ('row')

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i;

        // Create a unique seed for each work-item to ensure different random numbers.
        // The base_seed should be changed for each kernel call (e.g., based on time or iteration).
        uint seed = base_seed + weight_idx;

        // Simple Linear Congruential Generator (LCG) for pseudo-random numbers.
        seed = (seed * 1664525u + 1013904223u);
        float rand_val = (float)(seed) / (float)(0xFFFFFFFF); // Normalize to [0.0, 1.0]

        if (rand_val >= dropout_rate) {
            weights[weight_idx] -= learning_rate * gweights[weight_idx];
        }
        // If rand_val < dropout_rate, the weight is "dropped" and remains unchanged.
    }
}

inline void gradient_update(double *c, double *b, double x, double L, int n) {
    double factor = 1.0 - L * (n - 1.0) / x;
    double old_c = *c;
    *c = 0.9 * factor * old_c;
    *b = 0.1 * factor * old_c;
}

