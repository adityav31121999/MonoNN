#ifndef LOSS_HPP
#define LOSS_HPP 1
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <map>

#define LEARNING_MAX 0.01f          // maximum learning rate allowed
#define LEARNING_MIN 0.00001f       // minimum learning rate allowed
#define LAMBDA_L1 0.0001f           // L1 regularization parameter
#define LAMBDA_L2 0.0025f           // L2 regularization parameter
#define DROPOUT_RATE 0.6f           // dropout rate
#define DECAY_RATE 0.001f           // weight decay rate
#define WEIGHT_DECAY 0.001f         // weight decay parameter
#define SOFTMAX_TEMP 1.05f          // softmax temperature
#define EPOCH 100                   // epochs for single set training
#define SESSION_SIZE 10             // number of batches in single session
#define BATCH_SIZE 10               // number of inputs in single batch
#define ALPHA 0.85f                 // gradient splitting factor

// struct to hold statistical information about data
struct Statistics {
    float mean;     // mean value
    float std;      // standard deviation
    float min;      // minimum value from the set
    float max;      // maximum value from the set
};

Statistics computeStats(const std::vector<float>& data);
Statistics computeStats(const std::vector<std::vector<float>>& data);
void computeStats(const std::vector<std::vector<std::vector<float>>>& cweights, const std::vector<std::vector<std::vector<float>>>& bweights,
        const std::vector<std::vector<std::vector<float>>>& cgrad, const std::vector<std::vector<std::vector<float>>>& bgrad,
        const std::vector<std::vector<float>>& act);
void computeStats(const std::vector<std::vector<std::vector<float>>>& cweights, const std::vector<std::vector<std::vector<float>>>& bweights,
        const std::vector<std::vector<std::vector<float>>>& cgrad, const std::vector<std::vector<std::vector<float>>>& bgrad,
        const std::vector<std::vector<std::vector<float>>>& act);
void computeStats(const std::vector<std::vector<std::vector<float>>>& cweights, const std::vector<std::vector<std::vector<float>>>& bweights,
        const std::vector<std::vector<std::vector<float>>>& cgrad, const std::vector<std::vector<std::vector<float>>>& bgrad,
        const std::vector<std::vector<std::vector<std::vector<float>>>>& act);

// struct to save and access information on testing and training of neural network
// single session will have fixed number of batches or files to be trained on
struct progress {
    // files
    unsigned int batchSize;             // number of files in single batch (1 or many)
    unsigned int sessionSize;           // number of batches to be trained in single session (1 or many)
    unsigned int totalTrainFiles;       // total training files
    unsigned int totalTestFiles;        // total test files

    // training
    float currentLearningRate;                  // current session's learning rate after successful training
    float loss;                                 // loss after successful training
    double accLoss;                             // accumulated loss till current session
    unsigned int filesProcessed;                // number of files processed in current session
    unsigned long long ongoingCycleCount;       // total cyles of trainig till previous successful training
    unsigned long long totalCycleCount;         // total cycles after full training
    unsigned int totalSessionsOfTraining;       // total sessions used for training
    double timeForCurrentSession;               // time taken for current session
    double timeTakenForTraining;                // total time taken throughout sessions

    // testing
    float testError;                            // error recorded during testing
    float testAccuracy;                         // accuracy recorded during testing (correct predictions / total testing files)
    unsigned int correctPredictions;            // correct predictions done in testing
};

bool logProgressToCSV(const progress& p, const std::string& filePath);
bool loadLastProgress(progress& p, const std::string& filePath);

// file operations for weights serialization

void makeBinFile(const std::string& fileAddress, unsigned long long param);
void serializeWeights(const std::vector<std::vector<std::vector<float>>>& cweights,
                        const std::vector<std::vector<std::vector<float>>>& bweights,
                        const std::string& fileAddress);
void deserializeWeights(std::vector<float>& cweights, std::vector<float>& bweights,
                        const std::string& fileAddress);
void deserializeWeights(std::vector<std::vector<std::vector<float>>>& cweights,
                        std::vector<std::vector<std::vector<float>>>& bweights,
                        const std::vector<int>& width, const std::vector<int>& height,
                        const std::string& fileAddress);

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
float categoricalCrossEntropy(const std::vector<std::vector<float>>& output, const std::vector<std::vector<float>>& target);

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
void clipGradients(std::vector<std::vector<float>>& gradients, float max_norm);
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

// batch layer forprop

void layerForwardBatch(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output,
                       const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights);
void layerForwardBatch(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output,
                       const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n);
void layerForwardBatch(const std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<std::vector<float>>>& output,
                       const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights);
void layerForwardBatch(const std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<std::vector<float>>>& output,
                       const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n);

// single layer backprop (with direct weights update)

void layerBackward(const std::vector<float>& incoming, const std::vector<float>& prevAct, std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb, float m, float alpha);
void layerBackward(const std::vector<float>& incoming, std::vector<float>& outgoing, const std::vector<float>& prevAct,
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb, float m, float alpha);
void layerBackward(const std::vector<std::vector<float>>& incoming, const std::vector<std::vector<float>>& prevAct,
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb, float m, float alpha);
void layerBackward(const std::vector<std::vector<float>>& incoming, std::vector<std::vector<float>>& outgoing,
                    const std::vector<std::vector<float>>& dotProds, const std::vector<std::vector<float>>& prevAct,
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb, float m, float alpha);

// batch layer backprop

void layerBackwardBatch(const std::vector<std::vector<float>>& incoming, const std::vector<std::vector<float>>& prevAct, 
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb,
                    float m, float alpha);
void layerBackwardBatch(const std::vector<std::vector<std::vector<float>>>& incoming, const std::vector<std::vector<std::vector<float>>>& prevAct,
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb, 
                    float m, float alpha);
void layerBackwardBatch(const std::vector<std::vector<float>>& incoming, std::vector<std::vector<float>>& outgoing,
                    const std::vector<std::vector<float>>& prevAct, std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb,
                    float m, float alpha);
void layerBackwardBatch(const std::vector<std::vector<std::vector<float>>>& incoming, std::vector<std::vector<std::vector<float>>>& outgoing,
                    const std::vector<std::vector<std::vector<float>>>& dotProds, const std::vector<std::vector<std::vector<float>>>& prevAct,
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb,
                    float m, float alpha);

// image access and manipulation
#define NOMINMAX
#include <opencv2/core.hpp>

std::vector<std::vector<float>> cvMat2vec(const cv::Mat& mat);
cv::Mat vec2cvMat(const std::vector<std::vector<float>>& vec);
cv::Mat image2grey(const std::string& path2image);
std::vector<std::vector<std::vector<float>>> image2channels(const std::string& path2image);

#ifdef USE_CL

// Conditional inclusion of OpenCL C++ header based on OS
#if defined(_WIN64)
    #define CL_HPP_TARGET_OPENCL_VERSION 300
    // For Windows, use the older/common cl.hpp
    #include <CL/cl.hpp>
#elif defined(__linux__)
    #define CL_HPP_TARGET_OPENCL_VERSION 300
    #include <CL/opencl.hpp>
#endif

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

    extern const std::vector<std::string> kernelFiles;
    extern const std::vector<std::string> kernelNames;
    void createKernelsFromFile(const cl::Context& context, const std::vector<std::string>& filePath, std::map<std::string, cl::Kernel>& kernelMap);

#define WORKSIZE_1D 256
#define WORKSIZE_2DX 16
#define WORKSIZE_2DY 16

inline auto calculate_global_1d = [](size_t local_work_size_1d, size_t total_size) { 
    return ((total_size + local_work_size_1d - 1) / local_work_size_1d) * local_work_size_1d; 
};

inline auto calculate_global_2d = [](size_t local_work_size_2d_arr[2], size_t dim0, size_t dim1) { 
    size_t g0 = ((dim0 + local_work_size_2d_arr[0] - 1) / local_work_size_2d_arr[0]) * local_work_size_2d_arr[0];
    size_t g1 = ((dim1 + local_work_size_2d_arr[1] - 1) / local_work_size_2d_arr[1]) * local_work_size_2d_arr[1];
    return cl::NDRange(g0, g1); 
};

// to be used in CL_CHECK macro
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

#elif USE_CU

#include <cuda_runtime.h> // For cudaError_t, cudaGetErrorString, etc.
#include <stdexcept>      // For std::runtime_error

// --- CUDA Error Checking Macro ---
#define CUDA_CHECK(call)                                                    \
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
extern "C" __global__ void kernelUpdateWeightsDropout(float* weights, float* gweights, int totalElements, float learning_rate,
                                         float dropout_rate, unsigned int base_seed);

#endif

#endif // LOSS_HPP