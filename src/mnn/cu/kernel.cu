#ifdef USE_CUDA
#include <cfloat> // For FLT_MAX
#include <cmath>  // For signbit, pow, exp
#include <cuda_runtime.h>
#include <cuda.h>

// Helper macro/function for parallel reduction within a thread block.
// This is a common pattern for block-level reductions.
// Note: This pattern assumes blockDim.x is a power of 2.
#define WORK_GROUP_REDUCE(OP, INIT_VAL) \
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) { \
        __syncthreads(); \
        if (threadIdx.x < s) { \
            local_buffer[threadIdx.x] = OP(local_buffer[threadIdx.x], local_buffer[threadIdx.x + s]); \
        } \
    }

// CUDA device functions must be prefixed with __device__

__device__ inline float OP_MAX(float a, float b) { return max(a, b); }
__device__ inline float OP_MIN(float a, float b) { return min(a, b); }
__device__ inline float OP_SUM(float a, float b) { return a + b; }
__device__ inline float OP_SUB(float a, float b) { return a - b; }
__device__ inline float OP_MUL(float a, float b) { return a * b; }
__device__ inline float OP_DIV(float a, float b) { return a / b; }
__device__ inline float OP_POW(float a, float b) { return pow(a, b); }
__device__ inline float OP_EXP(float a) { return exp(a); }
__device__ inline float OP_SIGN(float a) {
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
        out[i] = 1.0f / (1.0f + exp(-x[i]));
    }
}

extern "C" __global__ void sigmoidDer(float* x, float* out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float sigmoid_x = 1.0f / (1.0f + exp(-x[i]));
        out[i] = sigmoid_x * (1.0f - sigmoid_x);
    }
}

extern "C" __global__ void softmax(float* x, float* out, float temp, int size)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x; // For global data access (linear index)
    int local_id = threadIdx.x;                           // For shared memory access
    unsigned int local_size = blockDim.x;                 // Thread block size

    // Assuming a max block size of 256 for the shared buffer
    __shared__ float local_buffer[256];

    // --- Step 1: Find max value using parallel reduction ---

    // Load data into shared memory. Elements outside 'size' get a safe identity value.
    float my_val = (global_id < size) ? x[global_id] : -FLT_MAX;
    local_buffer[local_id] = my_val;

    __syncthreads(); // Wait for all threads to load their data

    // Perform parallel max reduction (using the defined macro)
    WORK_GROUP_REDUCE(OP_MAX, -FLT_MAX)

    // Thread 0 now has the max value for the block
    float max_val = local_buffer[0];

    __syncthreads(); // Wait for max_val to be visible to all threads

    // --- Step 2: Compute exponentials and sum using parallel reduction ---

    // Compute shifted exponential. Elements outside 'size' get 0.0f (identity for sum).
    float shifted_exp = 0.0f;
    if (global_id < size) {
        shifted_exp = exp((x[global_id] - max_val) / temp);
    }

    local_buffer[local_id] = shifted_exp;

    __syncthreads(); // Wait for all threads to compute exponentials

    // Perform parallel sum reduction
    WORK_GROUP_REDUCE(OP_SUM, 0.0f)

    // Thread 0 now has the sum of exponentials for the block
    float sum_val = local_buffer[0];

    __syncthreads(); // Wait for sum_val to be visible

    // --- Step 3: Normalize elements and write output ---
    if (global_id < size) {
        if (sum_val > 0.0f) {
            out[global_id] = shifted_exp / sum_val;
        } else {
            out[global_id] = 0.0f;
        }
    }
}


// Softmax Derivative Kernel
extern "C" __global__ void softmaxDer(float* x, float* out, float temp, int size)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;
    unsigned int local_size = blockDim.x;

    __shared__ float local_buffer[256]; // Must be large enough for max blockDim.x

    // --- Step 1 & 2: Identical to Softmax kernel to get max_val and sum_val ---

    float my_val = (global_id < size) ? x[global_id] : -FLT_MAX;
    local_buffer[local_id] = my_val;
    __syncthreads();
    WORK_GROUP_REDUCE(OP_MAX, -FLT_MAX)
    float max_val = local_buffer[0];
    __syncthreads();

    float shifted_exp = 0.0f;
    if (global_id < size) {
        shifted_exp = exp((x[global_id] - max_val) / temp);
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
        out[i] = pow(x[i], n);
    }
}

extern "C" __global__ void dPower(float* x, float* out, float n, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = n * pow(x[i], n - 1.0f);
    }
}

extern "C" __global__ void meanPool(float* in, float* out, int inRows, int inCols, int poolSize)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; // c is the column index.
    if (c < inCols) {
        float sum = 0.0f;
        for (int r = 0; r < inRows; ++r) {
            sum += in[r * inCols + c];
        }
        out[c] = sum / (float)inRows;
    }
}

extern "C" __global__ void maxPool(float* in, float* out, int inRows, int inCols, int poolSize) {
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
                max_val = max(max_val, in[in_row * inCols + in_col]);
            }
        }
        out[r * outCols + c] = max_val;
    }
}

extern "C" __global__ void transpose(const float* in, float* out, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {
        out[j * rows + i] = in[i * cols + j];
    }
}

extern "C" __global__ void vecxvec2vec(const float* x1, const float* x2, float* result, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] = x1[i] * x2[i];
    }
}

extern "C" __global__ void vecxvec2mat(const float* x1, const float* x2, float* result, int x1size, int x2size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int resultSize = x1size * x2size;
    if (i < resultSize) {
        int row = i / x2size;
        int col = i % x2size;
        result[i] = x1[row] * x2[col]; // Outer Product
    }
}

extern "C" __global__ void vecxmat2vec(const float* vec, const float* mat, float* result, int matRows, int matCols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // i is the column index of the result vector (0 to matCols-1)
    if (i < matCols) {
        float sum = 0.0f; // Initialize accumulator
        // j is the index of vec / row index of mat (0 to matRows-1)
        for (int j = 0; j < matRows; j++) {
            // mat[j][i] index is j * matCols + i
            sum += vec[j] * mat[j * matCols + i];
        }
        result[i] = sum;
    }
}

extern "C" __global__ void matxmat2mat(const float* mat1, const float* mat2,
                                       float* result, int mat1Rows, int mat1Cols, int mat2cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int resultSize = mat1Rows * mat2cols; // Total size of the result matrix (M x N)

    if (i < resultSize) {
        int r = i / mat2cols; // Row index in mat1 and result (M)
        int c = i % mat2cols; // Col index in mat2 and result (N)
        int K = mat1Cols;     // Inner dimension (K)

        float sum = 0.0f; // Initialize accumulator
        for (int k = 0; k < K; k++) {
            // result[r][c] = sum(mat1[r][k] * mat2[k][c])
            // mat1[r][k] index: r * mat1Cols + k
            // mat2[k][c] index: k * mat2cols + c
            sum += mat1[r * mat1Cols + k] * mat2[k * mat2cols + c];
        }
        result[i] = sum;
    }
}

extern "C" __global__ void matxvec2vec(const float* mat, const float* vec, float* result, int matRows, int matCols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // i is the row index of mat / element index of result (0 to matRows-1)
    if (i < matRows) {
        float sum = 0.0f; // Initialize accumulator
        // j is the column index of mat / element index of vec (0 to matCols-1)
        for (int j = 0; j < matCols; j++) {
            // mat[i][j] index: i * matCols + j
            sum += mat[i * matCols + j] * vec[j];
        }
        result[i] = sum;
    }
}

extern "C" __global__ void hadamard(const float* mat1, const float* mat2, float* result, int mat1Rows, int mat1Cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = mat1Rows * mat1Cols; // Total number of elements in the matrix

    if (i < totalSize) {
        // Element-wise multiplication (Hadamard product)
        result[i] = mat1[i] * mat2[i];
    }
}

extern "C" __global__ void hadamard2(const float* mat1, const float* mat2, const float* mat3,
                                     float* result, int mat1Rows, int mat1Cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = mat1Rows * mat1Cols; // Total number of elements in the matrix

    if (i < totalSize) {
        // Element-wise multiplication (Hadamard product)
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
    // Note: This remains a serial implementation on a single thread.
    // For large matrices, a parallel reduction would be more efficient.
    if ((blockIdx.x * blockDim.x + threadIdx.x) == 0 && (blockIdx.y * blockDim.y + threadIdx.y) == 0) {
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
            sum += (input[i] * cweights[i * outSize + j]) + bweights[i * outSize + j];
        }
        output[j] += sum;
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
            float powered_input_i = pow(input[i], n);
            sum += (powered_input_i * cweights[i * outSize + j]) + bweights[i * outSize + j];
        }
        output[j] += sum;
    }
}

extern "C" __global__ void kernelLayerForward3(const float* input, float* output,
                                               const float* cweights, const float* bweights,
                                               int inHeight, int inWidth, int outSize)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (i < inHeight && j < outSize) {
        float dotProd_ij = 0.0f;
        for (int k = 0; k < inWidth; ++k) {
            dotProd_ij += input[i * inWidth + k] * cweights[k * outSize + j];
        }
        output[i * outSize + j] = dotProd_ij + bweights[i * outSize + j];
    }
}

extern "C" __global__ void kernelLayerForward4(const float* input, float* output,
                                               const float* cweights, const float* bweights,
                                               int inHeight, int inWidth, int outSize, float n)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (i < inHeight && j < outSize) {
        float dotProd_ij = 0.0f;
        for (int k = 0; k < inWidth; ++k) {
            dotProd_ij += pow(input[i * inWidth + k], n) * cweights[k * outSize + j];
        }
        output[i * outSize + j] = dotProd_ij + bweights[i * outSize + j];
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

extern "C" __global__ void kernelUpdateWeightsDropout(float* weights,
                                                      float* gweights,
                                                      int totalElements,
                                                      float learning_rate,
                                                      float dropout_rate,
                                                      unsigned int base_seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalElements) {
        // Create a unique seed for each thread.
        // Note: For robust applications, using the cuRAND library is recommended.
        unsigned int seed = base_seed + i;

        // Simple Linear Congruential Generator (LCG) for pseudo-random numbers.
        seed = (seed * 1664525u + 1013904223u);
        float rand_val = (float)(seed) / (float)(0xFFFFFFFF); // Normalize to [0.0, 1.0]

        if (rand_val >= dropout_rate) {
            weights[i] -= learning_rate * gweights[i];
        }
        // If rand_val < dropout_rate, the weight is "dropped" and remains unchanged.
    }
}

#endif