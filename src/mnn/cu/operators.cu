#ifdef USE_CU
#include "operators.hpp"
#include <cfloat> // For FLT_MAX
#include <cmath>  // For signbit, pow, exp
#include <cuda_runtime.h>
#include "device_functions.cuh" // Include the new header for device functions
#include <cuda.h>

/// ----------------- Math Functions ----------------- ///

extern "C" __global__ void add(const float* x, const float* y, float* out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = x[i] + y[i];
    }
}

extern "C" __global__ void subtract(const float* x, const float* y, float* out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = x[i] - y[i];
    }
}

extern "C" __global__ void scaleByValue(const float* x, float* out, float val, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = x[i] * val;
    }
}

extern "C" __global__ void power(const float* x, float* out, float n, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = powf(x[i], n);
    }
}

extern "C" __global__ void dPower(const float* x, float* out, float n, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = n * powf(x[i], n - 1.0f);
    }
}

extern "C" __global__ void meanPool(const float* in, float* out, int inRows, int inCols, int poolSize)
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

extern "C" __global__ void maxPool(const float* in, float* out, int inRows, int inCols, int poolSize)
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
    int r = blockIdx.y * blockDim.y + threadIdx.y; // Row index in mat1 and result (M)
    int c = blockIdx.x * blockDim.x + threadIdx.x; // Col index in mat2 and result (N)

    if (r < mat1Rows && c < mat2cols) {
        int K = mat1Cols; // Inner dimension

        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += mat1[r * K + k] * mat2[k * mat2cols + c];
        }
        result[r * mat2cols + c] = sum;
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
    if (threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0) {
        float sum = 0.0f;
        for (int i = 0; i < Rows * Cols; ++i) {
            sum += inputBuffer[i];
        }
        outputBuffer[0] = sum;
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

#endif // USE_CU