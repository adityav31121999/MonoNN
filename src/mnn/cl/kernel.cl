/// ----------------- Forward Propagation ----------------- ///

__kernel void kernelLayerForward1(__global const float* input, __global float* output,
                                  __global const float* cweights, __global const float* bweights,
                                  int inSize, int outSize)
{
    // Each work-item computes one element of the output vector.
    int j = get_global_id(0); // Index for the output vector (column index for cweights/bweights)

    if (j < outSize) {
        float sum = 0.0f;
        for (int i = 0; i < inSize; ++i) {
            // cweights[i][j] -> cweights[i * outSize + j]
            // bweights[i][j] -> bweights[i * outSize + j]
            sum += (input[i] * cweights[i * outSize + j]) + bweights[i * outSize + j];
        }
        // The C++ layerForward uses `output[j] += ...`, implying accumulation.
        // We assume 'output' is pre-initialized (e.g., to zeros) on the host
        // or that this kernel is meant to accumulate into existing values.
        output[j] += sum;
    }
}

__kernel void kernelLayerForward2(__global const float* input, __global float* output,
                                  __global const float* cweights, __global const float* bweights,
                                  int inSize, int outSize, float n)
{
    // Each work-item computes one element of the output vector.
    int j = get_global_id(0); // Index for the output vector (column index for cweights/bweights)

    if (j < outSize) {
        float sum = 0.0f;
        for (int i = 0; i < inSize; ++i) {
            // Calculate powerIn[i] = pow(input[i], n)
            float powered_input_i = pow(input[i], n);
            // cweights[i][j] -> cweights[i * outSize + j]
            // bweights[i][j] -> bweights[i * outSize + j]
            sum += (powered_input_i * cweights[i * outSize + j]) + bweights[i * outSize + j];
        }
        // The C++ layerForward uses `output[j] += ...`, implying accumulation.
        // We assume 'output' is pre-initialized (e.g., to zeros) on the host
        // or that this kernel is meant to accumulate into existing values.
        output[j] += sum;
    }
}

__kernel void kernelLayerForward3(__global const float* input, __global float* output,
                                  __global const float* cweights, __global const float* bweights,
                                  int inHeight, int inWidth, int outSize)
{
    // Each work-item computes one element of the output matrix.
    // Global ID 0 for row index, Global ID 1 for column index.
    int i = get_global_id(0); // Row index for input, output, bweights
    int j = get_global_id(1); // Column index for output, cweights, bweights

    // The C++ code's error checks and usage for bweights are inconsistent.
    // Following the usage `output[i][j] = ... + bweights[i][j]`,
    // we assume bweights is an `inHeight x outSize` matrix.
    // cweights is `inWidth x outSize`.
    // input is `inHeight x inWidth`.
    // output is `inHeight x outSize`.

    if (i < inHeight && j < outSize) {
        float dotProd_ij = 0.0f;
        for (int k = 0; k < inWidth; ++k) { // k iterates over inWidth (cweights_rows)
            // input[i][k] -> input[i * inWidth + k]
            // cweights[k][j] -> cweights[k * outSize + j]
            dotProd_ij += input[i * inWidth + k] * cweights[k * outSize + j];
        }
        // bweights[i][j] -> bweights[i * outSize + j]
        // The C++ layerForward uses `output[i][j] = ...`, implying assignment.
        output[i * outSize + j] = dotProd_ij + bweights[i * outSize + j];
    }
}

__kernel void kernelLayerForward4(__global const float* input, __global float* output,
                                  __global const float* cweights, __global const float* bweights,
                                  int inHeight, int inWidth, int outSize, float n)
{
    // Each work-item computes one element of the output matrix.
    // Global ID 0 for row index, Global ID 1 for column index.
    int i = get_global_id(0); // Row index for input, output, bweights
    int j = get_global_id(1); // Column index for output, cweights, bweights

    // The C++ code's error checks and usage for bweights are inconsistent.
    // Following the usage `output[i][j] = ... + bweights[i][j]`,
    // we assume bweights is an `inHeight x outSize` matrix.
    // cweights is `inWidth x outSize`.
    // input is `inHeight x inWidth`.
    // output is `inHeight x outSize`.

    if (i < inHeight && j < outSize) {
        float dotProd_ij = 0.0f;
        for (int k = 0; k < inWidth; ++k) { // k iterates over inWidth (cweights_rows)
            // input[i][k] -> input[i * inWidth + k]
            // powerIn[i][k] is pow(input[i * inWidth + k], n)
            // cweights[k][j] -> cweights[k * outSize + j]
            dotProd_ij += pow(input[i * inWidth + k], n) * cweights[k * outSize + j];
        }
        // bweights[i][j] -> bweights[i * outSize + j]
        // The C++ layerForward uses `output[i][j] = ...`, implying assignment.
        output[i * outSize + j] = dotProd_ij + bweights[i * outSize + j];
    }
}

/// ----------------- Backpropagation ----------------- ///

inline void kernelGradient(double *c, double *b, double x, double L, int n) {
    double factor = 1.0 - L * (n - 1.0) / x;
    double old_c = *c;
    *c = 0.9 * factor * old_c;
    *b = 0.1 * factor * old_c;
}

/// ----------------- Update Weights ----------------- ///

__kernel void kernelUpdateWeights(__global float* weights, 
                                __global float* gweights,
                                float learning_rate,
                                int current_layer_size,
                                int prev_layer_size)
{
    int j = get_global_id(0);
    int i = get_global_id(1);

    if (j < current_layer_size && i < prev_layer_size) {
        int weight_idx = j * prev_layer_size + i; // Assuming row-major for weights
        // The original kernel had `gweights[weight_idx] = gradient;` which was incomplete.
        // Assuming gweights already contains the gradient for this weight.
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
