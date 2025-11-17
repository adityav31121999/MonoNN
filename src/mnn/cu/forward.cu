#ifdef USE_CU
#include "operators.hpp"
#include <cfloat> // For FLT_MAX
#include <cmath>  // For signbit, pow, exp
#include <cuda_runtime.h>
#include <cuda.h>

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

/// ----------------- Batch Forward Propagation ----------------- ///

extern "C" __global__ void kernelLayerForwardBatch1(const float* input, float* output,
                                  const float* cweights, const float* bweights,
                                  int batchSize, int inSize, int outSize)
{
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y; // Batch index
    int j = blockIdx.x * blockDim.x + threadIdx.x;         // Index for the output vector

    if (batch_idx < batchSize && j < outSize) {
        int input_offset = batch_idx * inSize;
        int output_offset = batch_idx * outSize;

        float sum = 0.0f;
        for (int i = 0; i < inSize; ++i) {
            int weight_idx = i * outSize + j;
            sum += (input[input_offset + i] * cweights[weight_idx]) + bweights[weight_idx];
        }

        float final_val = output[output_offset + j] + sum;

        if (isnan(final_val)) {
            final_val = 0.0f;
        }
        else if (isinf(final_val)) {
            final_val = 1.0f;
        }
        
        output[output_offset + j] = final_val;
    }
}

extern "C" __global__ void kernelLayerForwardBatch2(const float* input, float* output,
                                  const float* cweights, const float* bweights,
                                  int batchSize, int inSize, int outSize, float n) {
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y; // Batch index
    int j = blockIdx.x * blockDim.x + threadIdx.x;         // Index for the output vector

    if (batch_idx < batchSize && j < outSize) {
        int input_offset = batch_idx * inSize;
        int output_offset = batch_idx * outSize;

        float sum = 0.0f;
        for (int i = 0; i < inSize; ++i) {
            float powered_input_i = powf(input[input_offset + i], n);
            int weight_idx = i * outSize + j;
            sum += (powered_input_i * cweights[weight_idx]) + bweights[weight_idx];
        }

        float final_val = output[output_offset + j] + sum;

        if (isnan(final_val)) {
            final_val = 0.0f;
        }
        else if (isinf(final_val)) {
            final_val = 1.0f;
        }
        
        output[output_offset + j] = final_val;
    }
}

extern "C" __global__ void kernelLayerForwardBatch3(const float* input, float* output,
                                  const float* cweights, const float* bweights,
                                  int batchSize, int inHeight, int inWidth, int outSize) {
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z; // Batch index
    int i = blockIdx.y * blockDim.y + threadIdx.y;         // Row index for the output matrix (0 to inHeight-1)
    int j = blockIdx.x * blockDim.x + threadIdx.x;         // Column index for the output matrix (0 to outSize-1)

    if (batch_idx < batchSize && i < inHeight && j < outSize) {
        int input_batch_offset = batch_idx * inHeight * inWidth;
        int output_batch_offset = batch_idx * inHeight * outSize;

        float dotProd_ij = 0.0f;
        for (int k = 0; k < inWidth; ++k) {
            int input_idx = input_batch_offset + i * inWidth + k;
            int weight_idx = k * outSize + j;
            dotProd_ij += (input[input_idx] * cweights[weight_idx]) + bweights[weight_idx];
        }

        if (isnan(dotProd_ij)) {
            dotProd_ij = 0.0f;
        } else if (isinf(dotProd_ij)) {
            dotProd_ij = 1.0f;
        }

        output[output_batch_offset + i * outSize + j] = dotProd_ij;
    }
}


extern "C" __global__ void kernelLayerForwardBatch4(const float* input, float* output,
                                  const float* cweights, const float* bweights,
                                  int batchSize, int inHeight, int inWidth, int outSize, float n) 
{
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z; // Batch index
    int i = blockIdx.y * blockDim.y + threadIdx.y;         // Row index for the output matrix (0 to inHeight-1)
    int j = blockIdx.x * blockDim.x + threadIdx.x;         // Column index for the output matrix (0 to outSize-1)

    if (batch_idx < batchSize && i < inHeight && j < outSize) {
        int input_batch_offset = batch_idx * inHeight * inWidth;
        int output_batch_offset = batch_idx * inHeight * outSize;

        float dotProd_ij = 0.0f;
        for (int k = 0; k < inWidth; ++k) {
            int input_idx = input_batch_offset + i * inWidth + k;
            int weight_idx = k * outSize + j;
            dotProd_ij += (powf(input[input_idx], n) * cweights[weight_idx]) + bweights[weight_idx];
        }

        if (isnan(dotProd_ij)) {
            dotProd_ij = 0.0f;
        } else if (isinf(dotProd_ij)) {
            dotProd_ij = 1.0f;
        }

        output[output_batch_offset + i * outSize + j] = dotProd_ij;
    }
}

#endif // USE_CU