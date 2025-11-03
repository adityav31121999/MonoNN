#ifndef MNN_HPP
#define MNN_HPP
#include <vector>
#include <string>
#include <utility>
#include "activations.hpp"
#include "operators.hpp"

#define LEARNING_MAX 0.01f          // maximum learning rate allowed
#define LEARNING_MIN 0.00001f       // minimum learning rate allowed
#define LAMBDA_L1 0.001f            // L1 regularization parameter
#define LAMBDA_L2 0.001f            // L2 regularization parameter
#define DROPOUT_RATE 0.6f           // dropout rate
#define DECAY_RATE 0.001f           // weight decay rate
#define WEIGHT_DECAY 0.001f         // weight decay parameter

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

/**
 * @brief Class representing a Monomial Neural Network (MNN).
 *      - The monomial is of the form f(x) = c*(x^n) + b
 *          - n: order of monomials
 *          - x: input to monimial`d mlp
 *          - c: coefficient of x^n
 *          - b: constant
 *          - Both c and b are trainable parameters.
 */
class mnn {
private:
    float order;                // order of neurons
    int inSize;                 // input size
    int outSize;                // output size
    int layers;                 // number of hidden layers
    int batchSize;              // number of inputs for batch training
    int epochs;                 // number of epochs
    int iterations;             // number of iterations
    float alpha;                // gradient splitting factor
    float learningRate;         // learning rate
    float lambdaL1;             // L1 regularization parameter
    float lambdaL2;             // L2 regularization parameter
    float decayRate;            // weight decay rate
    float dropoutRate;          // dropout rate
    std::vector<int> width;     // width of each layer and subsequent layers height

    std::vector<float> input;       // input vector
    std::vector<float> output;      // output vector
    std::vector<float> target;      // target vector

    unsigned long long param;       // counter for iterations
    std::string binFileAddress;     // binary file address to save weights and biases
    FILE* binFile = nullptr;        // binary file pointer to read/write weights and biases

// store values for training

    std::vector<std::vector<std::vector<float>>> cweights;      // c-coefficients of the network
    std::vector<std::vector<std::vector<float>>> bweights;      // b-constants of the network
    std::vector<std::vector<std::vector<float>>> cgradients;    // gradients of c-coefficients of the network
    std::vector<std::vector<std::vector<float>>> bgradients;    // gradients of b-constants of the network
    std::vector<std::vector<float>> dotProds;       // (activation^n) * cweights + bweights = dot products
    std::vector<std::vector<float>> activate;       // activations of the network

public:

// constructors

    mnn() = default;
    mnn(int insize, int outsize, int layers, float order, std::string binFileAddress);
    mnn(int insize, int outsize, int dim, int layers, float order, std::string binFileAddress);
    mnn(int insize, int outsize, std::vector<int> width, float order, std::string binFileAddress);

    void setbatch(int batchsize) { this->batchSize = batchSize; }
    void setepochs(int epochs) { this->epochs = epochs; }
    void setiterations(int iterations) { this->iterations = iterations; }
    void setlearningrate(float learningrate) { this->learningRate = learningRate; }
    void setlambdal1(float lambdaL1) { this->lambdaL1 = lambdaL1; }
    void setlambdal2(float lambdaL2) { this->lambdaL2 = lambdaL2; }
    void setdecayrate(float decayRate) { this->decayRate = decayRate; }
    void setdropoutrate(float dropoutRate) { this->dropoutRate = dropoutRate; }
    void setCGradients(std::vector<std::vector<float>>& cgradient, int layerNumber) { cgradients[layerNumber - 1] = cgradient; }
    void setBGradients(std::vector<std::vector<float>>& bgradient, int layerNumber) { bgradients[layerNumber - 1] = bgradient; }
    std::vector<std::vector<float>> getCLayer(int layerNumber) { return cweights[layerNumber-1]; }
    std::vector<std::vector<float>> getBLayer(int layerNumber) { return bweights[layerNumber-1]; }
    void makeBinFile(const std::string& fileAddress);

    #ifdef USE_CPU
        void forprop(std::vector<float>& input);
        void backprop(std::vector<float>& target);
        void backprop(std::vector<std::vector<float>>& target);
        void train(const std::vector<float>& input, const std::vector<float>& target);
        void trainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets);
    #elif USE_CUDA
        void cuForprop(std::vector<float>& input);
        void cuBackprop(std::vector<float>& target);
        void cuBackprop(std::vector<std::vector<float>> target);
        void cuTrain(const std::vector<float>& input, const std::vector<float>& target);
        void cuTrainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets); 
    #elif USE_OPENCL
        void clForprop(std::vector<float>& input);
        void clBackprop(std::vector<float>& target);
        void clBackprop(std::vector<std::vector<float>> target);
        void clTrain(const std::vector<float>& input, const std::vector<float>& target);
        void clTrainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets); 
    #endif

    ~mnn() = default;
};


/**
 * @brief Class representing a Monomial Neural Network (MNN) for 2D input and 1D output.
 *      - The monomial is of the form f(x) = c*(x^n) + b
 *          - n: order of monomials
 *          - x: input to monimial`d mlp
 *          - c: coefficient of x^n
 *          - b: constant
 *          - Both c and b are trainable parameters.
 */
class mnn2d {
private:
    float order;                    // order of neurons
    int inWidth;                    // input matrix width
    int inHeight;                   // input matrix height
    int outWidth;                   // output matrix width
    int layers;                     // number of hidden layers
    int batchSize;                  // batch size for training
    int epochs;                     // number of epochs
    int iterations;                 // number of iterations
    float alpha;                    // gradient splitting factor
    float learningRate;             // learning rate
    float lambdaL1;                 // L1 regularization parameter
    float lambdaL2;                 // L2 regularization parameter
    float decayRate;                // weight decay rate
    float dropoutRate;              // dropout rate
    std::vector<int> width;         // width of each layer

    std::vector<std::vector<float>> input;      // input vector
    std::vector<float> output;      // output vector: Mean Pool activate[layers-1]
    std::vector<float> target;      // target vector

    unsigned long long param;       // counter for iterations
    std::string binFileAddress;     // binary file address to save weights and biases
    FILE* binFile;                  // binary file pointer to read/write weights and biases

// weights and biases

    std::vector<std::vector<std::vector<float>>> cweights;      // c-coefficients of the network
    std::vector<std::vector<std::vector<float>>> bweights;      // b-constants of the network
    std::vector<std::vector<std::vector<float>>> cgradients;    // gradients of c-coefficients of the network
    std::vector<std::vector<std::vector<float>>> bgradients;    // gradients of b-constants of the network
    std::vector<std::vector<std::vector<float>>> dotProds;      // (activation^n) * clayers + blayers = dot products
    std::vector<std::vector<std::vector<float>>> activate;      // activations of the network

public:
// constructors

    mnn2d() = default;
    mnn2d(int inw, int inh, int outw, int layers, float order, std::string binFileAddress);
    mnn2d(int inw, int inh, int outw, int dim, int layers, float order, std::string binFileAddress);
    mnn2d(int inw, int inh, int outw, std::vector<int> width, float order, std::string binFileAddress);

    void setbatch(int batchsize) { this->batchSize = batchsize; }
    void setepochs(int epochs) { this->epochs = epochs; }
    void setiterations(int iterations) { this->iterations = iterations; }
    void setlearningrate(float learningrate) { this->learningRate = learningrate; }
    void setlambdal1(float lambdaL1) { this->lambdaL1 = lambdaL1; }
    void setlambdal2(float lambdaL2) { this->lambdaL2 = lambdaL2; }
    void setdecayrate(float decayRate) { this->decayRate = decayRate; }
    void setdropoutrate(float dropoutRate) { this->dropoutRate = dropoutRate; }
    void setCGradients(std::vector<std::vector<float>>& cgradient, int layerNumber) { cgradients[layerNumber - 1] = cgradient; }
    void setBGradients(std::vector<std::vector<float>>& bgradient, int layerNumber) { bgradients[layerNumber - 1] = bgradient; }
    std::vector<std::vector<float>> getCLayer(int layerNumber) { return cweights[layerNumber-1]; }
    std::vector<std::vector<float>> getBLayer(int layerNumber) { return bweights[layerNumber-1]; }
    void makeBinFile(const std::string& fileAddress);

    #ifdef USE_CPU
        void forprop(std::vector<std::vector<float>>& input);
        void backprop(std::vector<float>& target);
        void backprop(std::vector<std::vector<float>>& target);
        void train(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void trainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);
    #elif USE_CUDA
        void cuForprop(std::vector<std::vector<float>>& input);
        void cuBackprop(std::vector<float> target);
        void cuBackprop(std::vector<std::vector<float>> target);
        void cuTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void cuTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);
    #elif USE_OPENCL
        void clForprop(std::vector<std::vector<float>>& input);
        void clBackprop(std::vector<float> target);
        void clBackprop(std::vector<std::vector<float>> target);
        void clTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void clTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);
    #endif

    ~mnn2d() = default;
};

#ifdef USE_OPENCL
    std::vector<std::string> kernelNames = {
        // forward propagation kernels
        "kernelLayerForward1",
        "kernelLayerForward2",
        "kernelLayerForward3",
        "kernelLayerForward4",
        // backpropagation kernels

        // weight update kernels
        "kernelUpdateWeights",
        "kernelUpdateWeightsWithL1",
        "kernelUpdateWeightsWithL2",
        "kernelUpdateWeightsElasticNet"
        "kernelUpdateWeightsWeightDecay",
        "kernelUpdateWeightsDropout"
    };
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

#endif // MNN_HPP