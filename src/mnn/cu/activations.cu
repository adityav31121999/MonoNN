#ifdef USE_CU
#include <cfloat> // For FLT_MAX
#include <cmath>  // For signbit, pow, exp
#include <cuda_runtime.h>
#include "operators.hpp"
#include "device_functions.cuh" // Include the new header for device functions

// Helper macro/function for parallel reduction within a thread block.

/// ----------------- Activations and their Derivatives ----------------- ///

extern "C" __global__ void sigmoid(float* x, float* out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

extern "C" __global__ void sigmoidDer(float* x, float* out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float sigmoid_x = 1.0f / (1.0f + expf(-x[i]));
        out[i] = sigmoid_x * (1.0f - sigmoid_x);
    }
}

extern "C" __global__ void softmax_reduce(const float* input, float* partial_results,
                                         float* local_max_dummy, // Dummy, not used
                                         float* local_sum_dummy, // Dummy, not used
                                         int size, float temp)
{
    extern __shared__ float shared_mem[];       // dynamic memory, divided into two part
    float* local_max = shared_mem;
    float* local_sum = &shared_mem[blockDim.x];

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;
    int group_id = blockIdx.x;
    int local_size = blockDim.x;
    int global_size = gridDim.x * blockDim.x;

    // maximim value in work group
    float my_max = -FLT_MAX;
    for (int i = global_id; i < size; i += global_size) {
        my_max = fmaxf(my_max, input[i]);
    }
    local_max[local_id] = my_max;

    __syncthreads();

    // Parallel reduction to find the single max for the entire work-group
    for (int offset = local_size / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            local_max[local_id] = fmaxf(local_max[local_id], local_max[local_id + offset]);
        }
        __syncthreads();
    }

    //sum of exponentials
    float group_max = local_max[0];
    float my_sum = 0.0f;
    for (int i = global_id; i < size; i += global_size) {
        my_sum += expf((input[i] - group_max) / temp);
    }
    local_sum[local_id] = my_sum;

    __syncthreads();

    // Parallel reduction to find the single sum for the entire work-group
    for (int offset = local_size / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            local_sum[local_id] = local_sum[local_id] + local_sum[local_id + offset];
        }
        __syncthreads();
    }

    // write partial results
    if (local_id == 0) {
        partial_results[2 * group_id] = local_sum[0];
        partial_results[2 * group_id + 1] = group_max;
    }
}


extern "C" __global__ void softmax_normalize(const float* input,
                                float* output,
                                int size,
                                float temp,
                                float global_max, // The final max value
                                float global_sum) // The final sum value
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id < size) {
        float val = expf((input[global_id] - global_max) / temp);
        output[global_id] = val / global_sum;
    }
}


extern "C" __global__ void softmaxDer_normalize(const float* input,
                                   float* output,
                                   int size,
                                   float temp,
                                   float global_max,
                                   float global_sum)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id < size) {
        float val = expf((input[global_id] - global_max) / temp);
        float s_i = val / global_sum;
        // The derivative is s_i * (1 - s_i)
        output[global_id] = s_i * (1.0f - s_i);
    }
}

extern "C" __global__ void softmax(float* x, float* out, float temp, int size)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    __shared__ float local_buffer[256];

    // --- Step 1: Find max value using parallel reduction ---
    float my_val = (global_id < size) ? x[global_id] : -FLT_MAX;
    local_buffer[local_id] = my_val;

    __syncthreads();

    // Perform parallel max reduction
    WORK_GROUP_REDUCE(OP_MAX, -FLT_MAX)

    float max_val = local_buffer[0];
    __syncthreads();

    // --- Step 2: Compute exponentials and sum using parallel reduction ---
    float shifted_exp = 0.0f;
    if (global_id < size) {
        shifted_exp = expf((x[global_id] - max_val) / temp);
    }
    local_buffer[local_id] = shifted_exp;

    __syncthreads();

    // Perform parallel sum reduction
    WORK_GROUP_REDUCE(OP_SUM, 0.0f)

    float sum_val = local_buffer[0];
    __syncthreads();

    // --- Step 3: Normalize elements and write output ---
    if (global_id < size) {
        if (sum_val > 0.0f) {
            out[global_id] = shifted_exp / sum_val;
        } else {
            out[global_id] = 0.0f;
        }
    }
}

extern "C" __global__ void softmaxDer(float* x, float* out, float temp, int size)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    __shared__ float local_buffer[256];

    // --- Step 1 & 2: Identical to Softmax kernel to get max_val and sum_val ---
    float my_val = (global_id < size) ? x[global_id] : -FLT_MAX;
    local_buffer[local_id] = my_val;
    __syncthreads();
    WORK_GROUP_REDUCE(OP_MAX, -FLT_MAX)
    float max_val = local_buffer[0];
    __syncthreads();

    float shifted_exp = 0.0f;
    if (global_id < size) {
        shifted_exp = expf((x[global_id] - max_val) / temp);
    }
    local_buffer[local_id] = shifted_exp;
    __syncthreads();
    WORK_GROUP_REDUCE(OP_SUM, 0.0f)
    float sum_val = local_buffer[0];
    __syncthreads();

    // --- Step 3: Calculate Softmax and its derivative ---
    if (global_id < size) {
        float s_i = 0.0f;
        if (sum_val > 0.0f) {
            s_i = shifted_exp / sum_val;
        }
        out[global_id] = s_i * (1.0f - s_i);
    }
}

#endif // USE_CU