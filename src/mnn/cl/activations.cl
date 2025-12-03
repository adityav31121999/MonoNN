// Helper macro/function for parallel reduction within a work-group.
// This is a common pattern for work-group reductions.
// Note: This pattern assumes local_size is a power of 2.
#define WORK_GROUP_REDUCE(OP, INIT_VAL) \
    for (uint s = local_size / 2; s > 0; s /= 2) { \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (local_id < s) { \
            local_buffer[local_id] = OP(local_buffer[local_id], local_buffer[local_id + s]); \
        } \
    }

#ifndef FLT_MAX
    #define FLT_MAX 3.402823466e+38f
#endif

inline float OP_MAX(float a, float b) { return max(a, b); }
inline float OP_MIN(float a, float b) { return min(a, b); }
inline float OP_SUM(float a, float b) { return a + b; }
inline float OP_SUB(float a, float b) { return a - b; }
inline float OP_MUL(float a, float b) { return a * b; }
inline float OP_DIV(float a, float b) { return a / b; }
inline float OP_POW(float a, float b) { return pow(a, b); }
inline float OP_EXP(float a) { return exp(a); }
inline float OP_SIGN(float a) { 
    if (a == 0.0f)
        return 0.0f;
    else
        return signbit(a) ? -1.0f : 1.0f;
}
inline float valueCorrection(float a) {
    if(isnan(a)) {
        return 0.0f;
    }
    else if (isinf(a)) {
        return OP_SIGN(a);
    }
    else {
        return a;
    }
}

/// ----------------- Activations and their Derivatives ----------------- ///

__kernel void sigmoid(__global float* x, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        float val = 1.0f / (1.0f + exp(-x[i]));
        out[i] = val;
    }
}

__kernel void sigmoidDer(__global float* x, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        float sigmoid_x = 1.0f / (1.0f + exp(-x[i]));
        float val = sigmoid_x * (1.0f - sigmoid_x);
        out[i] = val;
    }
}

/*
 * KERNEL 1: REDUCTION STEP
 * This kernel finds the max value and the sum of exponentials.
 * It takes a large input array and produces a small output array containing
 * the partial results from each work-group.
 */
__kernel void softmax_reduce(__global const float* input,
                             __global float* partial_results, // Intermediate buffer
                             __local float* shared_mem,        // dynamic memory
                             int size,
                             float temp)
{
    __local float* local_max_ptr = shared_mem;
    __local float* local_sum = &shared_mem[get_local_size(0)];

    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);
    int global_size = get_global_size(0);

    // --- Part 1: Find the max value in the work-group ---
    float my_max = -FLT_MAX;
    for (int i = global_id; i < size; i += global_size) {
        my_max = max(my_max, input[i]);
    }
    local_max_ptr[local_id] = my_max;

    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction to find the single max for the entire work-group
    for (int offset = local_size / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            local_max_ptr[local_id] = max(local_max_ptr[local_id], local_max_ptr[local_id + offset]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // The group's max value is now in local_max[0]

    // --- Part 2: Find the sum of exponentials in the work-group ---
    float group_max = local_max_ptr[0];
    float my_sum = 0.0f;
    for (int i = global_id; i < size; i += global_size) {
        my_sum += exp((input[i] - group_max) / temp);
    }
    local_sum[local_id] = my_sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction to find the single sum for the entire work-group
    for (int offset = local_size / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            local_sum[local_id] = local_sum[local_id] + local_sum[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- Part 3: Write partial results ---
    // Each work-group writes its own max and sum to the intermediate buffer.
    if (local_id == 0) {
        partial_results[2 * group_id] = local_sum[0];
        partial_results[2 * group_id + 1] = group_max;
    }
}


/*
 * KERNEL 2: NORMALIZATION STEP
 * This is a simple kernel that applies the final global max and sum
 * to normalize every element of the input array.
 */
__kernel void softmax_normalize(__global const float* input,
                                __global float* output,
                                int size,
                                float temp,
                                float global_max, // The final max value
                                float global_sum) // The final sum value
{
    int global_id = get_global_id(0);

    if (global_id < size) {
        float val = exp((input[global_id] - global_max) / temp);
        // float corrected_val = valueCorrection(val);
        output[global_id] = val / global_sum;
    }
}

/*
 * KERNEL 3: SOFTMAX DERIVATIVE NORMALIZATION STEP
 * This kernel calculates the derivative s_i * (1 - s_i) after computing
 * the softmax value s_i using the global max and sum.
 */
__kernel void softmaxDer_normalize(__global const float* input,
                                   __global float* output,
                                   int size,
                                   float temp,
                                   float global_max,
                                   float global_sum)
{
    int global_id = get_global_id(0);

    if (global_id < size) {
        float val = exp((input[global_id] - global_max) / temp);
        float s_i = val / global_sum;
        // The derivative is s_i * (1 - s_i)
        output[global_id] = s_i * (1.0f - s_i);
    }
}

__kernel void softmax(__global float* x, __global float* out, float temp, int size)
{
    int global_id = get_global_id(0); // For global data access (linear index)
    int local_id = get_local_id(0);   // For local memory access
    uint local_size = get_local_size(0); // Work-group size

    // If the entire array must be processed by ONE work-group, 
    // the max local_size (e.g., 256 or 1024) must be >= size.
    // Error handling for size > max_local_size is typically done by the host.
    
    // Using a predefined size for local memory is necessary for OpenCL. 
    // This size must be at least 'local_size' (max expected WGS, e.g., 256).
    // The OpenCL 1.2 standard defines FLT_MAX for the maximum float.

    // Assuming a max work-group size of 256 for the local buffer
    __local float local_buffer[256]; 

    // --- Step 1: Find max value using parallel reduction ---

    // Load data into local memory. Elements outside 'size' get a safe identity value.
    float my_val = (global_id < size) ? x[global_id] : -FLT_MAX; 
    local_buffer[local_id] = my_val;

    barrier(CLK_LOCAL_MEM_FENCE); // Wait for all threads to load their data

    // Perform parallel max reduction (using the defined macro)
    WORK_GROUP_REDUCE(OP_MAX, -FLT_MAX)

    // Thread 0 now has the max value for the work-group
    float max_val = local_buffer[0];

    barrier(CLK_LOCAL_MEM_FENCE); // Wait for max_val to be visible to all threads

    // --- Step 2: Compute exponentials and sum using parallel reduction ---

    // Compute shifted exponential. Elements outside 'size' get 0.0f (identity for sum).
    float shifted_exp = 0.0f;
    if (global_id < size) {
        shifted_exp = exp((x[global_id] - max_val) / temp); 
    }
    
    local_buffer[local_id] = shifted_exp;

    barrier(CLK_LOCAL_MEM_FENCE); // Wait for all threads to compute exponentials

    // Perform parallel sum reduction
    WORK_GROUP_REDUCE(OP_SUM, 0.0f)

    // Thread 0 now has the sum of exponentials for the work-group
    float sum_val = local_buffer[0];

    barrier(CLK_LOCAL_MEM_FENCE); // Wait for sum_val to be visible

    // --- Step 3: Normalize elements and write output ---
    if (global_id < size) {
        if (sum_val > 0.0f) {
            float val = shifted_exp / sum_val;
            out[global_id] = val;
        }
        else {
            // Handle sum == 0 case (should ideally not happen with max-shift unless underflow is severe).
            // Setting to a small positive value (e.g., 1/size) or 0.0f is implementation specific.
            // Setting to 0.0f is numerically safer if the result must be 0-1. 
            out[global_id] = 0.0f; 
        }
    }
}


// Softmax Derivative Kernel
// This calculates s_i * (1 - s_i) where s_i is the Softmax output.
__kernel void softmaxDer(__global float* x, __global float* out, float temp, int size)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    uint local_size = get_local_size(0);

    __local float local_buffer[256]; // Must be large enough for max local_size

    // --- Step 1 & 2: Identical to Softmax kernel to get max_val and sum_val ---
    
    float my_val = (global_id < size) ? x[global_id] : -FLT_MAX;
    local_buffer[local_id] = my_val;
    barrier(CLK_LOCAL_MEM_FENCE);
    WORK_GROUP_REDUCE(OP_MAX, -FLT_MAX)
    float max_val = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float shifted_exp = 0.0f;
    if (global_id < size) {
        shifted_exp = exp((x[global_id] - max_val) / temp);
    }
    
    local_buffer[local_id] = shifted_exp;
    barrier(CLK_LOCAL_MEM_FENCE);
    WORK_GROUP_REDUCE(OP_SUM, 0.0f)
    float sum_val = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Step 3: Calculate Softmax and its derivative ---
    if (global_id < size) {
        float s_i = 0.0f;
        if (sum_val > 0.0f) {
            s_i = shifted_exp / sum_val;
        } 
        
        // The derivative for a single element s_i with respect to its own input x_i is s_i * (1 - s_i).
        // This derivative is WRONG if x is the result of a previous operation and we're propagating the gradient.
        // It's correct if the derivative is with respect to the pre-softmax input (for one-hot targets).
        out[global_id] = s_i * (1.0f - s_i);
    }
}
