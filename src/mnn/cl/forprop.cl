/// ----------------- Forward Propagation ----------------- ///

__kernel void kernelLayerForward1(__global const float* input, __global float* output,
                                  __global const float* cweights, __global const float* bweights,
                                  int inSize, int outSize)
{
    // Each work-item computes one element of the output vector.
    int j = get_global_id(0); // Index for the output vector

    if (j < outSize) {
        float sum = 0.0f;
        for (int i = 0; i < inSize; ++i) {
            // Index for cweights[i][j] and bweights[i][j]
            int weight_idx = i * outSize + j;
            sum += (input[i] * cweights[weight_idx]) + bweights[weight_idx];
        }

        // Accumulate into the existing output value, as per the C++ code's `output[j] += ...`
        float final_val = output[j] + sum;

        // Check for NaN and infinity, as done in the C++ code
        if (isnan(final_val)) {
            final_val = 0.0f;
        } else if (isinf(final_val)) {
            final_val = 1.0f;
        }
        
        output[j] = final_val;
    }
}

__kernel void kernelLayerForward2(__global const float* input, __global float* output,
                                  __global const float* cweights, __global const float* bweights,
                                  int inSize, int outSize, float n)
{
    // Each work-item computes one element of the output vector.
    int j = get_global_id(0); // Index for the output vector

    if (j < outSize) {
        float sum = 0.0f;
        for (int i = 0; i < inSize; ++i) {
            float powered_input_i = pow(input[i], n);
            // Index for cweights[i][j] and bweights[i][j]
            int weight_idx = i * outSize + j;
            sum += (powered_input_i * cweights[weight_idx]) + bweights[weight_idx];
        }

        // Accumulate into the existing output value
        float final_val = output[j] + sum;

        // Check for NaN and infinity
        if (isnan(final_val)) {
            final_val = 0.0f;
        } else if (isinf(final_val)) {
            final_val = 1.0f;
        }
        
        output[j] = final_val;
    }
}

__kernel void kernelLayerForward3(__global const float* input, __global float* output,
                                  __global const float* cweights, __global const float* bweights,
                                  int inHeight, int inWidth, int outSize)
{
    // Each work-item computes one element of the output matrix.
    int i = get_global_id(0); // Row index (0 to inHeight-1)
    int j = get_global_id(1); // Column index (0 to outSize-1)

    if (i < inHeight && j < outSize) {
        float dotProd_ij = 0.0f;
        for (int k = 0; k < inWidth; ++k) {
            dotProd_ij += (input[i * inWidth + k] * cweights[k * outSize + j]) + bweights[k * outSize + j];
        }

        // Check for NaN and infinity
        if (isnan(dotProd_ij)) {
            dotProd_ij = 0.0f;
        } else if (isinf(dotProd_ij)) {
            dotProd_ij = 1.0f;
        }

        output[i * outSize + j] = dotProd_ij;
    }
}

__kernel void kernelLayerForward4(__global const float* input, __global float* output,
                                  __global const float* cweights, __global const float* bweights,
                                  int inHeight, int inWidth, int outSize, float n)
{
    // Each work-item computes one element of the output matrix.
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < inHeight && j < outSize) {
        float dotProd_ij = 0.0f;
        for (int k = 0; k < inWidth; ++k) {
            dotProd_ij += (pow(input[i * inWidth + k], n) * cweights[k * outSize + j]) + bweights[k * outSize + j];
        }

        // Check for NaN and infinity
        if (isnan(dotProd_ij)) {
            dotProd_ij = 0.0f;
        } else if (isinf(dotProd_ij)) {
            dotProd_ij = 1.0f;
        }

        output[i * outSize + j] = dotProd_ij;
    }
}

/// ----------------- Batch Forward Propagation ----------------- ///

__kernel void kernelLayerForwardBatch1(__global const float* input, __global float* output,
                                  __global const float* cweights, __global const float* bweights,
                                  int batchSize, int inSize, int outSize)
{
    // Each work-item computes one element of the output for one batch item.
    int batch_idx = get_global_id(0); // Batch index
    int j = get_global_id(1);         // Index for the output vector

    if (batch_idx < batchSize && j < outSize) {
        int input_offset = batch_idx * inSize;
        int output_offset = batch_idx * outSize;

        float sum = 0.0f;
        for (int i = 0; i < inSize; ++i) {
            // Index for cweights[i][j] and bweights[i][j]
            int weight_idx = i * outSize + j;
            sum += (input[input_offset + i] * cweights[weight_idx]) + bweights[weight_idx];
        }

        // Accumulate into the existing output value, as per the C++ code's `output[j] += ...`
        float final_val = output[output_offset + j] + sum;

        // Check for NaN and infinity, as done in the C++ code
        if (isnan(final_val)) {
            final_val = 0.0f;
        }
        else if (isinf(final_val)) {
            final_val = 1.0f;
        }
        
        output[output_offset + j] = final_val;
    }
}

__kernel void kernelLayerForwardBatch2(__global const float* input, __global float* output,
                                  __global const float* cweights, __global const float* bweights,
                                  int batchSize, int inSize, int outSize, float n) {
    // Each work-item computes one element of the output for one batch item.
    int batch_idx = get_global_id(0); // Batch index
    int j = get_global_id(1);         // Index for the output vector

    if (batch_idx < batchSize && j < outSize) {
        int input_offset = batch_idx * inSize;
        int output_offset = batch_idx * outSize;

        float sum = 0.0f;
        for (int i = 0; i < inSize; ++i) {
            float powered_input_i = pow(input[input_offset + i], n);
            // Index for cweights[i][j] and bweights[i][j]
            int weight_idx = i * outSize + j;
            sum += (powered_input_i * cweights[weight_idx]) + bweights[weight_idx];
        }

        // Accumulate into the existing output value
        float final_val = output[output_offset + j] + sum;

        // Check for NaN and infinity
        if (isnan(final_val)) {
            final_val = 0.0f;
        }
        else if (isinf(final_val)) {
            final_val = 1.0f;
        }
        
        output[output_offset + j] = final_val;
    }
}

__kernel void kernelLayerForwardBatch3(__global const float* input, __global float* output,
                                  __global const float* cweights, __global const float* bweights,
                                  int batchSize, int inHeight, int inWidth, int outSize) {
    // Each work-item computes one element of one output matrix in the batch.
    int batch_idx = get_global_id(0); // Batch index
    int i = get_global_id(1);         // Row index for the output matrix (0 to inHeight-1)
    int j = get_global_id(2);         // Column index for the output matrix (0 to outSize-1)

    if (batch_idx < batchSize && i < inHeight && j < outSize) {
        // Calculate the base offset for the current input and output matrices in the batch
        int input_batch_offset = batch_idx * inHeight * inWidth;
        int output_batch_offset = batch_idx * inHeight * outSize;

        float dotProd_ij = 0.0f;
        for (int k = 0; k < inWidth; ++k) {
            // Index for input: input[batch_idx][i][k]
            int input_idx = input_batch_offset + i * inWidth + k;
            // Index for weights: cweights[k][j] and bweights[k][j]
            int weight_idx = k * outSize + j;
            dotProd_ij += (input[input_idx] * cweights[weight_idx]) + bweights[weight_idx];
        }

        // Check for NaN and infinity
        if (isnan(dotProd_ij)) {
            dotProd_ij = 0.0f;
        } else if (isinf(dotProd_ij)) {
            dotProd_ij = 1.0f;
        }

        // Index for output: output[batch_idx][i][j]
        output[output_batch_offset + i * outSize + j] = dotProd_ij;
    }
}

__kernel void kernelLayerForwardBatch4(__global const float* input, __global float* output,
                                  __global const float* cweights, __global const float* bweights,
                                  int batchSize, int inHeight, int inWidth, int outSize, float n) {
    // Each work-item computes one element of one output matrix in the batch.
    int batch_idx = get_global_id(0); // Batch index
    int i = get_global_id(1);         // Row index for the output matrix (0 to inHeight-1)
    int j = get_global_id(2);         // Column index for the output matrix (0 to outSize-1)

    if (batch_idx < batchSize && i < inHeight && j < outSize) {
        int input_batch_offset = batch_idx * inHeight * inWidth;
        int output_batch_offset = batch_idx * inHeight * outSize;

        float dotProd_ij = 0.0f;
        for (int k = 0; k < inWidth; ++k) {
            int input_idx = input_batch_offset + i * inWidth + k;
            int weight_idx = k * outSize + j;
            dotProd_ij += (pow(input[input_idx], n) * cweights[weight_idx]) + bweights[weight_idx];
        }

        // Check for NaN and infinity
        if (isnan(dotProd_ij)) {
            dotProd_ij = 0.0f;
        } else if (isinf(dotProd_ij)) {
            dotProd_ij = 1.0f;
        }

        output[output_batch_offset + i * outSize + j] = dotProd_ij;
    }
}