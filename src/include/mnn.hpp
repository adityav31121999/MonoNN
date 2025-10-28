#ifndef MNN_HPP
#define MNN_HPP
#include <vector>
#include <string>
#include "activations.hpp"
#include "loss.hpp"

// necessary operators and functions

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
std::vector<float> weightedMeanPool(const std::vector<std::vector<float>>& input, const std::vector<float>& weights);
std::vector<float> flatten(const std::vector<std::vector<float>>& input);
std::vector<std::vector<float>> reshape(const std::vector<float>& input, int rows, int cols);
std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& input);

void setWeightsByNormalDist(std::vector<std::vector<std::vector<float>>>& weights, float mean, float stddev);
void setWeightsByUniformDist(std::vector<std::vector<std::vector<float>>>& weights, float lower, float upper);
void setWeightsByXavier(std::vector<std::vector<std::vector<float>>>& weights, int fin, int fout, bool uniformOrNot);
void setWeightsByHe(std::vector<std::vector<std::vector<float>>>& weights, int fin, int fout);
void setWeightsByLeCunn(std::vector<std::vector<std::vector<float>>>& weights, int fin, int fout);

void updateWeights(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float& learningRate);
void updateWeightsL1(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learningRate, float lambdaL1);
void updateWeightsL2(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learningRate, float lambdaL2);
void updateWeightsElastic(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learningRate, float lambdaL1, float lambdaL2);
void updateWeightsWeightDecay(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learningRate, float decayRate);
void updateWeightsDropout(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learning, float dropoutRate);

void layerForward(const std::vector<float>& input, std::vector<float>& output, const std::vector<std::vector<float>>& cweights,
                    const std::vector<std::vector<float>>& bweights);
void layerForward(const std::vector<float>& input, std::vector<float>& output, const std::vector<std::vector<float>>& cweights,
                    const std::vector<std::vector<float>>& bweights, float n);
void layerForward(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output, 
                    const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights);
void layerForward(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output, 
                    const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n);

#include <utility> // if needed for pair, but using refs

auto gradient_update = [](double& c, double& b, double x, double L, int n) {
    double factor = 1.0 - L * (n - 1.0) / x;
    double old_c = c;
    c = 0.9 * factor * old_c;
    b = 0.1 * factor * old_c;
};


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
    int batchSize;              // batch size for training
    int epochs;                 // number of epochs
    int iterations;             // number of iterations
    float learningRate;         // learning rate
    float lambdaL1;             // L1 regularization parameter
    float lambdaL2;             // L2 regularization parameter
    float decayRate;            // weight decay rate
    float dropoutRate;          // dropout rate
    std::vector<int> width;     // width of each layer and subsequent layers height

    std::vector<float> input;       // input vector
    std::vector<float> output;      // output vector
    std::vector<float> target;      // target vector

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
    mnn(int insize, int outsize, int layers, float order);
    mnn(int insize, int outsize, int dim, int layers, float order);
    mnn(int insize, int outsize, std::vector<int> width, float order);

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

    #ifdef USE_CPU
        void forprop(std::vector<float>& input);
        void backprop(std::vector<float>& target);
        void train(const std::vector<float>& input, const std::vector<float>& target);
        void trainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets);
    #elif USE_CUDA
        void cuForprop(std::vector<float>& input);
        void cuBackprop(std::vector<float>& target);
        void cuTrain(const std::vector<float>& input, const std::vector<float>& target);
        void cuTrainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets); 
    #elif USE_OPENCL
        void clForprop(std::vector<float>& input);
        void clBackprop(std::vector<float>& target);
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
    float learningRate;             // learning rate
    float lambdaL1;                 // L1 regularization parameter
    float lambdaL2;                 // L2 regularization parameter
    float decayRate;                // weight decay rate
    float dropoutRate;              // dropout rate
    std::vector<int> width;         // width of each layer

    std::vector<std::vector<float>> input;      // input vector
    std::vector<float> output;      // output vector: Mean Pool activate[layers-1]
    std::vector<float> target;      // target vector

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
    mnn2d(int inw, int inh, int outw, int layers, float order);
    mnn2d(int inw, int inh, int outw, int dim, int layers, float order);
    mnn2d(int inw, int inh, int outw, std::vector<int> width, float order);

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

    #ifdef USE_CPU
        void forprop(std::vector<std::vector<float>>& input);
        void backprop(std::vector<float> target);
        void train(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void trainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);
    #elif USE_CUDA
        void cuForprop(std::vector<std::vector<float>>& input);
        void cuBackprop(std::vector<float> target);
        void cuTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void cuTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);
    #elif USE_OPENCL
        void clForprop(std::vector<std::vector<float>>& input);
        void clBackprop(std::vector<float> target);
        void clTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void clTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);
    #endif

    ~mnn2d() = default;
};

#ifdef USE_OPENCL
    std::vector<std::string> kernelNames = {
        "kernelUpdateWeights",
        "kernelUpdateWeightsWithL1",
        "kernelUpdateWeightsWithL2",
        "kernelUpdateWeightsElasticNet"
        "kernelUpdateWeightsWeightDecay",
        "kernelUpdateWeightsDropout"
    };
#elif USE_CUDA
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