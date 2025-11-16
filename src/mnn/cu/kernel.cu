#ifdef USE_CU
#include <cfloat> // For FLT_MAX
#include <cmath>  // For signbit, pow, exp
#include <cuda_runtime.h>
#include <cuda.h>

#include <cfloat> // For FLT_MAX

// Helper macro/function for parallel reduction within a thread block.
#define WORK_GROUP_REDUCE(OP, INIT_VAL) \
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) { \
        __syncthreads(); \
        if (threadIdx.x < s) { \
            local_buffer[threadIdx.x] = OP(local_buffer[threadIdx.x], local_buffer[threadIdx.x + s]); \
        } \
    }

// CUDA device-side math functions
__device__ __forceinline__ float OP_MAX(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ float OP_MIN(float a, float b) { return fminf(a, b); }
__device__ __forceinline__ float OP_SUM(float a, float b) { return a + b; }
__device__ __forceinline__ float OP_SUB(float a, float b) { return a - b; }
__device__ __forceinline__ float OP_MUL(float a, float b) { return a * b; }
__device__ __forceinline__ float OP_DIV(float a, float b) { return a / b; }
__device__ __forceinline__ float OP_POW(float a, float b) { return powf(a, b); }
__device__ __forceinline__ float OP_EXP(float a) { return expf(a); }
__device__ __forceinline__ float OP_SIGN(float a) {
    if (a == 0.0f)
        return 0.0f;
    else
        return signbit(a) ? -1.0f : 1.0f;
}

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


/// ----------------- Math Functions ----------------- ///

extern "C" __global__ void add(float* x, float* y, float* out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = x[i] + y[i];
    }
}

extern "C" __global__ void subtract(float* x, float* y, float* out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = x[i] - y[i];
    }
}

extern "C" __global__ void scaleByValue(float* x, float* out, float val, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = x[i] * val;
    }
}

extern "C" __global__ void power(float* x, float* out, float n, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = powf(x[i], n);
    }
}

extern "C" __global__ void dPower(float* x, float* out, float n, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = n * powf(x[i], n - 1.0f);
    }
}

extern "C" __global__ void meanPool(float* in, float* out, int inRows, int inCols, int poolSize)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < inCols) {
        float sum = 0.0f;
        for (int r = 0; r < inRows; ++r) {
            sum += in[r * inCols + c];
        }
        out[c] = sum / (float)inRows;
    }
}

extern "C" __global__ void maxPool(float* in, float* out, int inRows, int inCols, int poolSize)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    int outRows = inRows / poolSize;
    int outCols = inCols / poolSize;

    if (r < outRows && c < outCols) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                int in_row = r * poolSize + i;
                int in_col = c * poolSize + j;
                max_val = fmaxf(max_val, in[in_row * inCols + in_col]);
            }
        }
        out[r * outCols + c] = max_val;
    }
}

extern "C" __global__ void transpose(const float* in, float* out, int rows, int cols)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {
        out[j * rows + i] = in[i * cols + j];
    }
}

extern "C" __global__ void vecxvec2vec(const float* x1, const float* x2, float* result, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] = x1[i] * x2[i];
    }
}

extern "C" __global__ void vecxvec2mat(const float* x1, const float* x2, float* result, int x1size, int x2size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int resultSize = x1size * x2size;
    if (i < resultSize) {
        int row = i / x2size;
        int col = i % x2size;
        result[i] = x1[row] * x2[col]; // Outer Product
    }
}

extern "C" __global__ void vecxmat2vec(const float* vec, const float* mat, float* result, int matRows, int matCols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < matCols) {
        float sum = 0.0f;
        for (int j = 0; j < matRows; j++) {
            sum += vec[j] * mat[j * matCols + i];
        }
        result[i] = sum;
    }
}

extern "C" __global__ void matxmat2mat(const float* mat1, const float* mat2,
                            float* result, int mat1Rows, int mat1Cols, int mat2cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; // Column of result matrix
    int r = blockIdx.y * blockDim.y + threadIdx.y; // Row of result matrix

    if (r < mat1Rows && c < mat2cols) {
        int i = r * mat2cols + c; // Linear index for the result matrix
        int K = mat1Cols; // Inner dimension for multiplication

        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += mat1[r * mat1Cols + k] * mat2[k * mat2cols + c];
        }
        result[i] = sum;
    }
}

extern "C" __global__ void matxvec2vec(const float* mat, const float* vec, float* result, int matRows, int matCols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < matRows) {
        float sum = 0.0f;
        for (int j = 0; j < matCols; j++) {
            sum += mat[i * matCols + j] * vec[j];
        }
        result[i] = sum;
    }
}

extern "C" __global__ void fill(float* out, float val, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        out[i] = val;
    }
}

extern "C" __global__ void hadamard(const float* mat1, const float* mat2, float* result, int mat1Rows, int mat1Cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = mat1Rows * mat1Cols;

    if (i < totalSize) {
        result[i] = mat1[i] * mat2[i];
    }
}

extern "C" __global__ void hadamard2(const float* mat1, const float* mat2, const float* mat3,
        float* result, int mat1Rows, int mat1Cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = mat1Rows * mat1Cols;

    if (i < totalSize) {
        result[i] = mat1[i] * mat2[i] * mat3[i];
    }
}

extern "C" __global__ void matrix_vector_average(
    const float* inputBuffer,
    float* outputBuffer,
    const int N,
    const int Rows,
    const int Cols)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (j >= Rows || k >= Cols) {
        return;
    }

    float sum = 0.0f;
    int matrixSize = Rows * Cols;

    for (int i = 0; i < N; ++i) {
        int inputIndex = i * matrixSize + j * Cols + k;
        sum += inputBuffer[inputIndex];
    }

    int outputIndex = j * Cols + k;
    outputBuffer[outputIndex] = sum / (float)N;
}

extern "C" __global__ void matrix_vector_sum(
    const float* inputBuffer,
    float* outputBuffer,
    const int Rows,
    const int Cols)
{
    // This is a simple, non-parallelized sum for demonstration.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < Rows * Cols; ++i) {
            sum += inputBuffer[i];
        }
        outputBuffer[0] = sum;
    }
}

/// ----------------- Forward Propagation ----------------- ///

extern "C" __global__ void kernelLayerForward1(const float* input, float* output,
                                  const float* cweights, const float* bweights,
                                  int inSize, int outSize)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < outSize) {
        float sum = 0.0f;
        for (int i = 0; i < inSize; ++i) {
            int weight_idx = i * outSize + j;
            sum += (input[i] * cweights[weight_idx]) + bweights[weight_idx];
        }

        float final_val = output[j] + sum;

        if (isnan(final_val)) {
            final_val = 0.0f;
        } else if (isinf(final_val)) {
            final_val = 1.0f;
        }
        
        output[j] = final_val;
    }
}

extern "C" __global__ void kernelLayerForward2(const float* input, float* output,
                                  const float* cweights, const float* bweights,
                                  int inSize, int outSize, float n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < outSize) {
        float sum = 0.0f;
        for (int i = 0; i < inSize; ++i) {
            float powered_input_i = powf(input[i], n);
            int weight_idx = i * outSize + j;
            sum += (powered_input_i * cweights[weight_idx]) + bweights[weight_idx];
        }

        float final_val = output[j] + sum;

        if (isnan(final_val)) {
            final_val = 0.0f;
        } else if (isinf(final_val)) {
            final_val = 1.0f;
        }
        
        output[j] = final_val;
    }
}

extern "C" __global__ void kernelLayerForward3(const float* input, float* output,
                                  const float* cweights, const float* bweights,
                                  int inHeight, int inWidth, int outSize)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < inHeight && j < outSize) {
        float dotProd_ij = 0.0f;
        for (int k = 0; k < inWidth; ++k) {
            dotProd_ij += (input[i * inWidth + k] * cweights[k * outSize + j]) + bweights[k * outSize + j];
        }

        if (isnan(dotProd_ij)) {
            dotProd_ij = 0.0f;
        } else if (isinf(dotProd_ij)) {
            dotProd_ij = 1.0f;
        }

        output[i * outSize + j] = dotProd_ij;
    }
}

extern "C" __global__ void kernelLayerForward4(const float* input, float* output,
                                  const float* cweights, const float* bweights,
                                  int inHeight, int inWidth, int outSize, float n)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < inHeight && j < outSize) {
        float dotProd_ij = 0.0f;
        for (int k = 0; k < inWidth; ++k) {
            dotProd_ij += (powf(input[i * inWidth + k], n) * cweights[k * outSize + j]) + bweights[k * outSize + j];
        }

        if (isnan(dotProd_ij)) {
            dotProd_ij = 0.0f;
        } else if (isinf(dotProd_ij)) {
            dotProd_ij = 1.0f;
        }

        output[i * outSize + j] = dotProd_ij;
    }
}

/// ----------------- Update Weights ----------------- ///

extern "C" __global__ void kernelUpdateWeights(float* weights,
                                float* gweights,
                                float learning_rate,
                                int totalElements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalElements) {
        weights[i] -= learning_rate * gweights[i];
    }
}

extern "C" __global__ void kernelUpdateWeightsWithL1(float* weights,
                                        float* gweights,
                                        int totalElements,
                                        float learning_rate,
                                        float lambda_l1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalElements) {
        float current_weight = weights[i];
        float sign = OP_SIGN(current_weight);
        float l1_gradient = sign * lambda_l1;
        float total_gradient = gweights[i] + l1_gradient;
        weights[i] -= learning_rate * total_gradient;
    }
}

extern "C" __global__ void kernelUpdateWeightsWithL2(float* weights,
                                        float* gweights,
                                        int totalElements,
                                        float learning_rate,
                                        float lambda_l2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalElements) {
        float current_weight = weights[i];
        float l2_gradient = lambda_l2 * current_weight;
        float total_gradient = gweights[i] + l2_gradient;
        weights[i] -= learning_rate * total_gradient;
    }
}

extern "C" __global__ void kernelUpdateWeightsElasticNet(float* weights,
                                            float* gweights,
                                            int totalElements,
                                            float learning_rate,
                                            float lambda_l1,
                                            float lambda_l2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalElements) {
        float current_weight = weights[i];
        float l2_gradient = lambda_l2 * current_weight;
        float sign = OP_SIGN(current_weight);
        float l1_gradient = sign * lambda_l1;
        float total_gradient = gweights[i] + l2_gradient + l1_gradient;
        weights[i] -= learning_rate * total_gradient;
    }
}

extern "C" __global__ void kernelUpdateWeightsWithWeightDecay(float* weights,
                                                 float* gweights,
                                                 int totalElements,
                                                 float learning_rate,
                                                 float decay_rate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalElements) {
        float current_weight = weights[i];
        float gradient = gweights[i];
        weights[i] = current_weight * (1.0f - decay_rate) - learning_rate * gradient;
    }
}

// NOTE: For robust random number generation in CUDA, the cuRAND library is recommended.
// This is a direct translation of the simple LCG for demonstration.
extern "C" __global__ void kernelUpdateWeightsDropout(float* weights,
                                         float* gweights,
                                         int totalElements,
                                         float learning_rate,
                                         float dropout_rate,
                                         unsigned int base_seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalElements) {
        unsigned int seed = base_seed + i;
        seed = (seed * 1664525u + 1013904223u);
        float rand_val = (float)(seed) / (float)(0xFFFFFFFF);

        if (rand_val >= dropout_rate) {
            weights[i] -= learning_rate * gweights[i];
        }
    }
}

#endif