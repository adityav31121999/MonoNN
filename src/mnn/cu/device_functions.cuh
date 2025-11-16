#ifndef MNN_CU_DEVICE_FUNCTIONS_CUH
#define MNN_CU_DEVICE_FUNCTIONS_CUH

#include <cmath> // For signbit, powf, expf
#include <cuda_runtime.h> // For __device__


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
__device__ __forceinline__ float OP_SIGN(float a) 
                                    {
                                        if (a == 0.0f)
                                            return 0.0f;
                                        else
                                            return signbit(a) ? -1.0f : 1.0f;
                                    }

#endif // MNN_CU_DEVICE_FUNCTIONS_CUH