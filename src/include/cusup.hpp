#ifndef CUSUP_HPP
#define CUSUP_HPP 1
#ifdef USE_CU

#include <cuda_runtime.h>
#include <stdexcept>
#include <curand_kernel.h>

// --- CUDA Error Checking Macro ---
#define CU_CHECK(call)                                                          \
    do {                                                                        \
        cudaError_t err_code_ = call;                                           \
        if (err_code_ != cudaSuccess) {                                         \
            std::string error_message_ = "CUDA API Error in ";                  \
            error_message_ += __FILE__;                                         \
            error_message_ += " at line " + std::to_string(__LINE__) + ": ";    \
            error_message_ += cudaGetErrorString(err_code_);                    \
            error_message_ += " (" + std::to_string(err_code_) + ")";           \
            throw std::runtime_error(error_message_);                           \
        }                                                                       \
    } while (0)


// Standard block sizes for kernel launches
constexpr int WORKSIZE_1D = 256;
constexpr int WORKSIZE_2D_X = 16;
constexpr int WORKSIZE_2D_Y = 16;

// Function to calculate grid size for 1D operations
inline dim3 calculate_grid_1d(int total_size, int block_size) {
    return dim3((total_size + block_size - 1) / block_size);
}

// Function to calculate grid size for 2D operations (dim_x = cols, dim_y = rows)
inline dim3 calculate_grid_2d(int dim_x, int dim_y, int block_x, int block_y) {
    return dim3((dim_x + block_x - 1) / block_x, (dim_y + block_y - 1) / block_y);
}

// actvations and derivative
extern "C" __global__ void sigmoid(const float* x, float* out, int size);
extern "C" __global__ void sigmoidDer(const float* x, float* out, int size);
extern "C" __global__ void relu(const float* x, float* out, int size);
extern "C" __global__ void reluDer(const float* x, float* out, int size);
extern "C" __global__ void softmax_reduce(const float* input, float* partial_results, int size, float temp);
extern "C" __global__ void softmax_normalize(const float* input, float* output, int size, float temp, float global_max, float global_sum);
extern "C" __global__ void softmaxDer_normalize(const float* input, float* output, int size, float temp, float global_max, float global_sum);
extern "C" __global__ void softmax(const float* x, float* out, float temp, int size);
extern "C" __global__ void softmaxDer(const float* x, float* out, float temp, int size);
// maths
extern "C" __global__ void add(const float* x, const float* y, float* out, int size);
extern "C" __global__ void subtract(const float* x, const float* y, float* out, int size);
extern "C" __global__ void scaleByValue(const float* x, float* out, float val, int size);
extern "C" __global__ void power(const float* x, float* out, float n, int size);
extern "C" __global__ void dPower(const float* x, float* out, float n, int size);
extern "C" __global__ void meanPool(const float* in, float* out, int inRows, int inCols, int poolSize);
extern "C" __global__ void maxPool(const float* in, float* out, int inRows, int inCols, int poolSize);
extern "C" __global__ void transpose(const float* in, float* out, int rows, int cols);
extern "C" __global__ void vecxvec2vec(const float* x1, const float* x2, float* result, int size);
extern "C" __global__ void vecxvec2mat(const float* x1, const float* x2, float* result, int x1size, int x2size);
extern "C" __global__ void vecxmat2vec(const float* vec, const float* mat, float* result, int matRows, int matCols);
extern "C" __global__ void matxmat2mat(const float* mat1, const float* mat2, float* result, int mat1Rows, int mat1Cols, int mat2cols);
extern "C" __global__ void matxvec2vec(const float* mat, const float* vec, float* result, int matRows, int matCols);
extern "C" __global__ void hadamard(const float* mat1, const float* mat2, float* result, int mat1Rows, int mat1Cols);
extern "C" __global__ void hadamard2(const float* mat1, const float* mat2, const float* mat3, float* result, int mat1Rows, int mat1Cols);
extern "C" __global__ void matrix_vector_average(const float* inputBuffer, float* outputBuffer, const int N, const int Rows, const int Cols);
extern "C" __global__ void matrix_vector_sum(const float* inputBuffer, float* outputBuffer, const int Rows, const int Cols);
// forward layer
extern "C" __global__ void kernelLayerForward1(const float* input, float* output, const float* cweights, const float* bweights,
                                  int inSize, int outSize);
extern "C" __global__ void kernelLayerForward2(const float* input, float* output, const float* cweights, const float* bweights,
                                  int inSize, int outSize, float n);
extern "C" __global__ void kernelLayerForward3(const float* input, float* output, const float* cweights, const float* bweights,
                                  int inHeight, int inWidth, int outSize);
extern "C" __global__ void kernelLayerForward4(const float* input, float* output, const float* cweights, const float* bweights,
                                  int inHeight, int inWidth, int outSize, float n);
// forward layer batch
extern "C" __global__ void kernelLayerForwardBatch1(const float* input, float* output, const float* cweights, const float* bweights,
                                  int batchSize, int inSize, int outSize);
extern "C" __global__ void kernelLayerForwardBatch2(const float* input, float* output, const float* cweights, const float* bweights,
                                  int batchSize, int inSize, int outSize, float n);
extern "C" __global__ void kernelLayerForwardBatch3(const float* input, float* output, const float* cweights, const float* bweights,
                                  int batchSize, int inHeight, int inWidth, int outSize);
extern "C" __global__ void kernelLayerForwardBatch4(const float* input, float* output, const float* cweights, const float* bweights,
                                  int batchSize, int inHeight, int inWidth, int outSize, float n);
// update weights
extern "C" __global__ void kernelUpdateWeights(float* weights, float* gweights, float learning_rate,
                                int totalElements);
extern "C" __global__ void kernelUpdateWeightsWithL1(float* weights, float* gweights, int totalElements, float learning_rate, float lambda_l1);
extern "C" __global__ void kernelUpdateWeightsWithL2(float* weights, float* gweights, int totalElements, float learning_rate, float lambda_l2);
extern "C" __global__ void kernelUpdateWeightsElasticNet(float* weights, float* gweights, int totalElements, float learning_rate,
                                float lambda_l1, float lambda_l2);
extern "C" __global__ void kernelUpdateWeightsWithWeightDecay(float* weights, float* gweights, int totalElements, float learning_rate, float decay_rate);
extern "C" __global__ void kernelUpdateWeightsDropout(float* weights, float* gweights, int totalElements, float learning_rate, float dropout_rate,
                                unsigned int);

#endif // USE_CU
#endif // CUSUP_HPP