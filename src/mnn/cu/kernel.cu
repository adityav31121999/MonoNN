#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include "operators.hpp"
#include <cfloat> // For FLT_MAX

__device__ inline float op_sign(float a) {
    if (a == 0.0f) return 0.0f;
    // copysignf is a fast way to get the sign of 'a' onto 1.0f
    return copysignf(1.0f, a);
}

// Helper for work-group reductions
template <typename T, typename Op>
__device__ T work_group_reduce(T val, Op op, T init_val) {
    extern __shared__ T shared_buffer[];
    int local_id = threadIdx.x;
    shared_buffer[local_id] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            shared_buffer[local_id] = op(shared_buffer[local_id], shared_buffer[local_id + s]);
        }
        __syncthreads();
    }
    return shared_buffer[0];
}

__device__ float op_max(float a, float b) { return fmaxf(a, b); }
__device__ float op_sum(float a, float b) { return a + b; }

/// ----------------- Activations and Derivatives ----------------- ///

__global__ void sigmoid(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

__global__ void sigmoidDer(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sigmoid_x = 1.0f / (1.0f + expf(-x[idx]));
        out[idx] = sigmoid_x * (1.0f - sigmoid_x);
    }
}

__global__ void softmax(const float* x, float* out, float temp, int size) {
    extern __shared__ float local_buffer[];
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    // Step 1: Find max value
    float my_val = (global_id < size) ? x[global_id] : -FLT_MAX;
    local_buffer[local_id] = my_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            local_buffer[local_id] = fmaxf(local_buffer[local_id], local_buffer[local_id + s]);
        }
        __syncthreads();
    }
    float max_val = local_buffer[0];
    __syncthreads();

    // Step 2: Compute exponentials and sum
    float shifted_exp = 0.0f;
    if (global_id < size) {
        shifted_exp = expf((x[global_id] - max_val) / temp);
    }
    local_buffer[local_id] = shifted_exp;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            local_buffer[local_id] += local_buffer[local_id + s];
        }
        __syncthreads();
    }
    float sum_val = local_buffer[0];
    __syncthreads();

    // Step 3: Normalize
    if (global_id < size) {
        if (sum_val > 0.0f) {
            out[global_id] = shifted_exp / sum_val;
        } else {
            out[global_id] = 0.0f;
        }
    }
}

__global__ void softmaxDer(const float* x, float* out, float temp, int size) {
    extern __shared__ float local_buffer[];
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    // Steps 1 & 2: Re-calculate max_val and sum_val
    float my_val = (global_id < size) ? x[global_id] : -FLT_MAX;
    local_buffer[local_id] = my_val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) { if (local_id < s) local_buffer[local_id] = fmaxf(local_buffer[local_id], local_buffer[local_id + s]); __syncthreads(); }
    float max_val = local_buffer[0];
    __syncthreads();

    float shifted_exp = 0.0f;
    if (global_id < size) {
        shifted_exp = expf((x[global_id] - max_val) / temp);
    }
    local_buffer[local_id] = shifted_exp;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) { if (local_id < s) local_buffer[local_id] += local_buffer[local_id + s]; __syncthreads(); }
    float sum_val = local_buffer[0];
    __syncthreads();

    // Step 3: Calculate softmax and its derivative
    if (global_id < size) {
        float s_i = (sum_val > 0.0f) ? (shifted_exp / sum_val) : 0.0f;
        out[global_id] = s_i * (1.0f - s_i);
    }
}

/// ----------------- Math Operations ----------------- ///

__global__ void add(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void subtract(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] - b[idx];
    }
}

__global__ void scaleByValue(const float* x, float* out, float val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] * val;
    }
}

__global__ void power(const float* x, float* out, float n, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = powf(x[idx], n);
    }
}

__global__ void meanPool(const float* in, float* out, int inRows, int inCols, int poolSize) {
    int r = blockIdx.y * blockDim.y + threadIdx.y; // Current row in the output
    int c = blockIdx.x * blockDim.x + threadIdx.x; // Current column in the output

    int outRows = inRows / poolSize;
    int outCols = inCols / poolSize;

    if (r < outRows && c < outCols) {
        float sum = 0.0f;
        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                int in_row = r * poolSize + i;
                int in_col = c * poolSize + j;
                sum += in[in_row * inCols + in_col];
            }
        }
        out[r * outCols + c] = sum / (float)(poolSize * poolSize);
    }
}

__global__ void maxPool(const float* in, float* out, int inRows, int inCols, int poolSize) {
    int r = blockIdx.y * blockDim.y + threadIdx.y; // Current row in the output
    int c = blockIdx.x * blockDim.x + threadIdx.x; // Current column in the output

    int outRows = inRows / poolSize;
    int outCols = inCols / poolSize;

    if (r < outRows && c < outCols) {
        float max_val = -FLT_MAX; // Initialize with a very small number
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


__global__ void dPower(const float* x, float* out, float n, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = n * powf(x[idx], n - 1.0f);
    }
}

__global__ void transpose(const float* in, float* out, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row in input, col in output
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col in input, row in output
    if (i < rows && j < cols) {
        out[j * rows + i] = in[i * cols + j];
    }
}

__global__ void hadamard(const float* mat1, const float* mat2, float* result, int mat1Rows, int mat1Cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = mat1Rows * mat1Cols;
    if (i < totalSize) {
        result[i] = mat1[i] * mat2[i];
    }
}

__global__ void hadamard2(const float* mat1, const float* mat2, const float* mat3, float* result, int mat1Rows, int mat1Cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int size = mat1Rows * mat1Cols;
    if (i < size) {
        result[i] = mat1[i] * mat2[i] * mat3[i];
    }
}

__global__ void vecxvec2vec(const float* x1, const float* x2, float* result, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] = x1[i] * x2[i];
    }
}

__global__ void vecxvec2mat(const float* x1, const float* x2, float* result, int x1size, int x2size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int resultSize = x1size * x2size;
    if (i < resultSize) {
        int row = i / x2size;
        int col = i % x2size;
        result[i] = x1[row] * x2[col]; // Outer Product
    }
}

__global__ void vecxmat2vec(const float* vec, const float* mat, float* result, int matRows, int matCols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // result col index
    if (i < matCols) {
        float sum = 0.0f;
        for (int j = 0; j < matRows; j++) {
            sum += vec[j] * mat[j * matCols + i];
        }
        result[i] = sum;
    }
}

__global__ void matxmat2mat(const float* mat1, const float* mat2, float* result, int mat1Rows, int mat1Cols, int mat2Cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y; // result row index
    int c = blockIdx.x * blockDim.x + threadIdx.x; // result col index

    if (r < mat1Rows && c < mat2Cols) {
        float sum = 0.0f;
        for (int k = 0; k < mat1Cols; k++) {
            sum += mat1[r * mat1Cols + k] * mat2[k * mat2Cols + c];
        }
        result[r * mat2Cols + c] = sum;
    }
}

__global__ void matxvec2vec(const float* mat, const float* vec, float* result, int matRows, int matCols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // result row index
    if (i < matRows) {
        float sum = 0.0f;
        for (int j = 0; j < matCols; j++) {
            sum += mat[i * matCols + j] * vec[j];
        }
        result[i] = sum;
    }
}

__global__ void matrix_vector_average(const float* inputBuffer, float* outputBuffer, int N, int Rows, int Cols) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (j < Rows && k < Cols) {
        float sum = 0.0f;
        int matrixSize = Rows * Cols;
        for (int i = 0; i < N; ++i) {
            sum += inputBuffer[i * matrixSize + j * Cols + k];
        }
        outputBuffer[j * Cols + k] = sum / (float)N;
    }
}

__global__ void matrix_vector_sum(const float* inputBuffer, float* outputBuffer, int Rows, int Cols) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < Rows * Cols; ++i) {
            sum += inputBuffer[i];
        }
        outputBuffer[0] = sum;
    }
}


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
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += (input[i] * weights[i * output_size + j]) + biases[i * output_size + j];
        }
        atomicAdd(&output[j], sum);
    }
}

__global__ void kernelLayerForward2(float* input,
                                    float* weights,
                                    float* biases,
                                    float* output,
                                    int input_size,
                                    int output_size,
                                    float n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // neuron index in current layer

    if (j < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            float powered_input_i = powf(input[i], n);
            sum += (powered_input_i * weights[i * output_size + j]) + biases[i * output_size + j];
        }
        atomicAdd(&output[j], sum);
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
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (i < inHeight && j < output_size) {
        float dotProd_ij = 0.0f;
        for (int k = 0; k < inWidth; ++k) {
            dotProd_ij += input[i * inWidth + k] * weights[k * output_size + j];
        }
        output[i * output_size + j] = dotProd_ij + biases[i * output_size + j];
    }
}


__global__ void kernelLayerForward4(float* input,
                                    float* weights,
                                    float* biases,
                                    float* output,
                                    int inHeight,
                                    int inWidth,
                                    int output_size,
                                    float n)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (i < inHeight && j < output_size) {
        float dotProd_ij = 0.0f;
        for (int k = 0; k < inWidth; ++k) {
            dotProd_ij += powf(input[i * inWidth + k], n) * weights[k * output_size + j];
        }
        output[i * output_size + j] = dotProd_ij + biases[i * output_size + j];
    }
}

/// ----------------- Backpropagation ----------------- ///

__global__ void kernelUpdateWeights(float* weights,
                                    float* gweights,
                                    float learning_rate,
                                    int totalElements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalElements) {
        weights[i] -= learning_rate * gweights[i];
    }
}

__global__ void kernelUpdateWeightsWithL1(float* weights,
                                          float* gweights,
                                          int totalElements,
                                          float learning_rate,
                                          float lambda_l1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalElements) {
        float l1_gradient = op_sign(weights[i]) * lambda_l1;
        float total_gradient = gweights[i] + l1_gradient;
        weights[i] -= learning_rate * total_gradient;
    }
}

__global__ void kernelUpdateWeightsWithL2(float* weights,
                                          float* gweights,
                                          int totalElements,
                                          float learning_rate,
                                          float lambda_l2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalElements) {
        float l2_gradient = lambda_l2 * weights[i];
        float total_gradient = gweights[i] + l2_gradient;
        weights[i] -= learning_rate * total_gradient;
    }
}

__global__ void kernelUpdateWeightsElasticNet(float* weights,
                                              float* gweights,
                                              float learning_rate,
                                              float lambda_l1,
                                              float lambda_l2,
                                              int totalElements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalElements) {
        float current_weight = weights[i];
        float l2_gradient = lambda_l2 * current_weight;
        float l1_gradient = op_sign(current_weight) * lambda_l1;
        float total_gradient = gweights[i] + l2_gradient + l1_gradient;
        weights[i] -= learning_rate * total_gradient;
    }
}

__global__ void kernelUpdateWeightsWithWeightDecay(float* weights,
                                                   float* gweights,
                                                   int totalElements,
                                                   float learning_rate,
                                                   float decay_rate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalElements) {
        weights[i] = weights[i] * (1.0f - decay_rate) - learning_rate * gweights[i];
    }
}

__global__ void kernelUpdateWeightsDropout(float* weights,
                                           float* gweights,
                                           float learning_rate,
                                           float dropout_rate,
                                           int totalElements,
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