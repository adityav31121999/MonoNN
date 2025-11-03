#ifndef LOSS_HPP
#define LOSS_HPP 1
#include <vector>
#include <cmath>
#include <string>
#include <map>

// activations and their derivatives

float sigmoid(float x);
float sigmoidDer(float x);
std::vector<float> sigmoid(const std::vector<float>& x);
std::vector<float> sigmoidDer(const std::vector<float>& x);
std::vector<std::vector<float>> sigmoid(const std::vector<std::vector<float>>& x);
std::vector<std::vector<float>> sigmoidDer(const std::vector<std::vector<float>>& x);

float relu(float x);
float reluDer(float x);
std::vector<float> relu(const std::vector<float>& x);
std::vector<float> reluDer(const std::vector<float>& x);
std::vector<std::vector<float>> relu(const std::vector<std::vector<float>>& x);
std::vector<std::vector<float>> reluDer(const std::vector<std::vector<float>>& x);

std::vector<float> softmax(const std::vector<float>& x);
std::vector<float> softmaxDer(const std::vector<float>& x);
std::vector<float> softmax(const std::vector<float>& x, float temp);
std::vector<float> softmaxDer(const std::vector<float>& x, float temp);

// errors

float mse(const std::vector<float>& output, const std::vector<float>& target);
float crossEntropy(const std::vector<float>& output, const std::vector<float>& target);
float binaryCrossEntropy(const std::vector<float>& output, const std::vector<float>& target);

// math operators

std::vector<float> operator+(const std::vector<float>& a, const std::vector<float>& b);
std::vector<std::vector<float>> operator+(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b);
std::vector<float> operator*(const std::vector<float>& a, const std::vector<std::vector<float>>& b);
std::vector<std::vector<float>> operator*(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b);
std::vector<float> multiply(const std::vector<float>& a, const std::vector<float>& b);
std::vector<float> multiply(const std::vector<float>& a, const std::vector<std::vector<float>>& b);
std::vector<std::vector<float>> multiply(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b);
std::vector<float> power(const std::vector<float>& input, const float& powerOfValues);
std::vector<std::vector<float>> power(const std::vector<std::vector<float>>& input, const float& powerOfValues);
std::vector<float> meanPool(const std::vector<std::vector<float>>& input);
std::vector<float> maxPool(const std::vector<std::vector<float>>& input);
std::vector<float> weightedMeanPool(const std::vector<float>& weights, const std::vector<std::vector<float>>& input);
std::vector<float> flatten(const std::vector<std::vector<float>>& input);
std::vector<std::vector<float>> reshape(const std::vector<float>& input, int rows, int cols);
std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& input);
std::vector<std::vector<float>> average(const std::vector<std::vector<std::vector<float>>>& input);
int maxIndex(const std::vector<float>& input);

// weight initialisation

void setWeightsByNormalDist(std::vector<std::vector<std::vector<float>>>& weights, float mean, float stddev);
void setWeightsByUniformDist(std::vector<std::vector<std::vector<float>>>& weights, float lower, float upper);
void setWeightsByXavier(std::vector<std::vector<std::vector<float>>>& weights, int fin, int fout, bool uniformOrNot);
void setWeightsByHe(std::vector<std::vector<std::vector<float>>>& weights, int fin, int fout);
void setWeightsByLeCunn(std::vector<std::vector<std::vector<float>>>& weights, int fin, int fout);

// modify weights with gradients

void updateWeights(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float& learningRate);
void updateWeightsL1(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learningRate, float lambdaL1);
void updateWeightsL2(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learningRate, float lambdaL2);
void updateWeightsElastic(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learningRate, float lambdaL1, float lambdaL2);
void updateWeightsWeightDecay(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learningRate, float decayRate);
void updateWeightsDropout(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learning, float dropoutRate);
void updateWeights(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float& learningRate, int type);

// single layer forprop

void layerForward(const std::vector<float>& input, std::vector<float>& output, const std::vector<std::vector<float>>& cweights,
                    const std::vector<std::vector<float>>& bweights);
void layerForward(const std::vector<float>& input, std::vector<float>& output, const std::vector<std::vector<float>>& cweights,
                    const std::vector<std::vector<float>>& bweights, float n);
void layerForward(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output, 
                    const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights);
void layerForward(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output, 
                    const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n);

// single layer backprop (with direct weights update)

void layerBackward(const std::vector<float>& incoming, const std::vector<float>& prevAct, 
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& B, 
                    std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb, 
                    float m, float alpha, float learning, int typeOfUpdate);
void layerBackward(const std::vector<float>& incoming, std::vector<float>& outgoing, const std::vector<float>& prevAct, 
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& B, 
                    std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb,
                    float m, float alpha, float learning, int typeOfUpdate);
void layerBackward(const std::vector<std::vector<float>>& incoming, const std::vector<std::vector<float>>& prevAct,
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& B,
                    std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb,
                    float m, float alpha, float learning, int typeOfUpdate);
void layerBackward(const std::vector<std::vector<float>>& incoming, std::vector<std::vector<float>>& outgoing,
                    const std::vector<std::vector<float>>& dotProds, const std::vector<std::vector<float>>& prevAct,
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& B,
                    std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb,
                    float m, float alpha, float learning, int typeOfUpdate);


#ifdef USE_OPENCL

// Conditional inclusion of OpenCL C++ header based on OS
#if defined(_WIN64)
    #define CL_HPP_ENABLE_EXCEPTIONS
    #define CL_HPP_TARGET_OPENCL_VERSION 200
    // For Windows, use the older/common cl.hpp
    #include <CL/cl.hpp>
#elif defined(__linux__)
    #define CL_HPP_TARGET_OPENCL_VERSION 220
    #define CL_TARGET_OPENCL_VERSION 220 // Inform C headers to target OpenCL 2.2
    #include <CL/opencl.hpp>
#endif

// --- OpenCL Error String Helper ---
// (Add this function definition before the CL_CHECK macro)
inline const char* oclErrorString(cl_int error) {
    switch (error) {
        // Run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // Compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
        // case -69: return "CL_INVALID_PIPE_SIZE";                 // OpenCL 2.0
        // case -70: return "CL_INVALID_DEVICE_QUEUE";              // OpenCL 2.0
        // case -71: return "CL_INVALID_SPEC_ID";                   // OpenCL 2.2
        // case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";     // OpenCL 2.2
        default: return "Unknown OpenCL error";
    }
}

#define CL_CHECK(call)                                                      \
    do {                                                                        \
        cl_int err_code_ = call;                                                \
        if (err_code_ != CL_SUCCESS) {                                          \
            std::string error_message_ = "OpenCL API Error in ";                \
            error_message_ += __FILE__;                                         \
            error_message_ += " at line " + std::to_string(__LINE__) + ": ";    \
            error_message_ += oclErrorString(err_code_);                        \
            error_message_ += " (" + std::to_string(err_code_) + ")";           \
            throw std::runtime_error(error_message_);                           \
        }                                                                       \
    } while (0)

    // Declare kernelNames as an external constant.
    // The actual definition will be in a .cpp file.
    extern const std::vector<std::string> kernelNames;
    void createKernelsFromFile(const cl::Context& context, const std::string& filePath, std::map<std::string, cl::Kernel>& kernelMap);

#elif USE_CUDA

    __global__ void kernelLayerForward1(float* input, float* weights, float* biases, float* output,
                                    int input_size, int output_size);
    __global__ void kernelLayerForward2(float* input, float* weights, float* biases, float* output,
                                    int input_size, int output_size, float n);
    __global__ void kernelLayerForward3(float* input, float* weights, float* biases, float* output,
                                    int inHeigt, int inWidth, int output_size);
    __global__ void kernelLayerForward4(float* input, float* weights, float* biases, float* output,
                                    int inHeigt, int inWidth, int output_size, float n);

    __global__ void kernelUpdateWeights(float* weights, float* gweights, float learning_rate,
                    int current_layer_size, int prev_layer_size);
    __global__ void kernelUpdateWeightsL1(float* weights, float* gweights, float learning_rate, float lambda_l1,
                    int current_layer_size, int prev_layer_size);
    __global__ void kernelUpdateWeightsL2(float* weights, float* gweights, float learning_rate,
                    float lambda_l2, int current_layer_size, int prev_layer_size);
    __global__ void kernelUpdateWeightsElasticNet(float* weights, float* gweights, float learning_rate, float lambda_l1,
                    float lambda_l2, int current_layer_size, int prev_layer_size);
    __global__ void kernelUpdateWeightsWeightDecay(float* weights, float* gweights, float learning_rate,
                    float decay_rate, int current_layer_size, int prev_layer_size);
    __global__ void kernelUpdateWeightsDropout(float* weights, float* gweights, float learning_rate, float dropout_rate,
                    int current_layer_size, int prev_layer_size);

#endif

#endif // LOSS_HPP