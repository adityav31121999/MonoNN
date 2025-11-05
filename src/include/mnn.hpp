#ifndef MNN_HPP
#define MNN_HPP 1
#include <vector>
#include <string>
#include <utility>
#include <map>
#include "operators.hpp"

#define LEARNING_MAX 0.01f          // maximum learning rate allowed
#define LEARNING_MIN 0.00001f       // minimum learning rate allowed
#define LAMBDA_L1 0.001f            // L1 regularization parameter
#define LAMBDA_L2 0.001f            // L2 regularization parameter
#define DROPOUT_RATE 0.6f           // dropout rate
#define DECAY_RATE 0.001f           // weight decay rate
#define WEIGHT_DECAY 0.001f         // weight decay parameter

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

    std::vector<std::vector<float>> inputBatch;     // input vector
    std::vector<std::vector<float>> outputBatch;    // output vector
    std::vector<std::vector<float>> targetBatch;    // target vector
    std::vector<std::vector<std::vector<float>>> dotProdsBatch; // for batch
    std::vector<std::vector<std::vector<float>>> activateBatch; // for batch

// constructors

    mnn() = default;
    mnn(int insize, int outsize, int layers, float order, std::string binFileAddress);
    mnn(int insize, int outsize, int dim, int layers, float order, std::string binFileAddress);
    mnn(int insize, int outsize, std::vector<int> width, float order, std::string binFileAddress);

    void setbatch(int batchsize) { this->batchSize = batchSize; }
    void setepochs(int epochs) { this->epochs = epochs; }
    void setiterations(int iterations) { this->iterations = iterations; }
    void setlearningrate(float learningrate) { this->learningRate = learningRate; }
    void setCWeights(std::vector<std::vector<float>>& cweight, int layerNumber) { cweights[layerNumber - 1] = cweight; }
    void setBWeights(std::vector<std::vector<float>>& bweight, int layerNumber) { bweights[layerNumber - 1] = bweight; }
    void setCGradients(std::vector<std::vector<float>>& cgradient, int layerNumber) { cgradients[layerNumber - 1] = cgradient; }
    void setBGradients(std::vector<std::vector<float>>& bgradient, int layerNumber) { bgradients[layerNumber - 1] = bgradient; }
    void setActivate(std::vector<float>& act, int layerNumber) { activate[layerNumber - 1] = act; }
    void setDotProds(std::vector<float>& dot, int layerNumber) { dotProds[layerNumber - 1] = dot; }
    // access C weight layer (1-based index)
    std::vector<std::vector<float>> getCWeights(int layerNumber) { return cweights[layerNumber-1]; }
    std::vector<std::vector<float>> getBWeights(int layerNumber) { return bweights[layerNumber-1]; }
    std::vector<std::vector<float>> getCGradients(int layerNumber) { return cgradients[layerNumber-1]; }
    std::vector<std::vector<float>> getBGradients(int layerNumber) { return bgradients[layerNumber-1]; }
    std::vector<float> getActivate(int layerNumber) { return activate[layerNumber-1]; }
    std::vector<float> getDotProds(int layerNumber) { return dotProds[layerNumber-1]; }
    // get rows and columns
    int getCWrows(int layerNumber) { return cweights[layerNumber].size(); }
    int getCWcols(int layerNumber) { return cweights[layerNumber][0].size(); }
    int getBWrows(int layerNumber) { return bweights[layerNumber].size(); }
    int getBWcols(int layerNumber) { return bweights[layerNumber][0].size(); }
    int getCGrows(int layerNumber) { return cgradients[layerNumber].size(); }
    int getCGcols(int layerNumber) { return cgradients[layerNumber][0].size(); }
    int getBGrows(int layerNumber) { return bgradients[layerNumber].size(); }
    int getBGcols(int layerNumber) { return bgradients[layerNumber][0].size(); }
    int getArows(int layerNumber) { return activate[layerNumber].size(); }
    int getDrows(int layerNumber) { return dotProds[layerNumber].size(); }

    void makeBinFile(const std::string& fileAddress);

    #ifdef USE_CPU

        void forprop(const std::vector<float>& input);
        void forprop(const std::vector<std::vector<float>>& input);
        void backprop(const std::vector<float>& target);
        void backprop(const std::vector<std::vector<float>>& target);
        void train(const std::vector<float>& input, const std::vector<float>& target);
        void trainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets);
        void thredTrain(const std::vector<float>& input, const std::vector<float>& target);
        void threadTrainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets);

    #elif USE_CUDA

        void cuForprop(const std::vector<float>& input);
        void cuBackprop(const std::vector<float>& target);
        void cuBackprop(const std::vector<std::vector<float>>& target);
        void cuTrain(const std::vector<float>& input, const std::vector<float>& target);
        void cuTrainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets); 
        void cuBufTrain(const std::vector<float>& input, const std::vector<float>& target);
        void cuBufTrainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets); 

    #elif USE_OPENCL

        cl::Context clContext;               // OpenCL context
        cl::CommandQueue clCommandQueue;     // OpenCL command queue
        cl::Device device;                  // Represents the selected OpenCL device.
        std::map<std::string, cl::Kernel> kernels; // Map to store kernel objects by name

        void clForprop(const std::vector<float>& input);
        void clBackprop(const std::vector<float>& target);
        void clBackprop(const std::vector<std::vector<float>>& target);
        void clTrain(const std::vector<float>& input, const std::vector<float>& target);
        void clTrainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets);
        void clBufTrain(const std::vector<float>& input, const std::vector<float>& target);
        void clBufTrainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets);

    #endif

    void train(const std::string& dataSetPath, int batchSize);
    void test(const std::string& dataSetPath, float loss);

// destructor
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

    std::vector<std::vector<std::vector<float>>> inputBatch;    // input vector
    std::vector<std::vector<float>> outputBatch;                // output vector
    std::vector<std::vector<float>> targetBatch;                // target vector
    std::vector<std::vector<std::vector<std::vector<float>>>> dotProdsBatch;    // for batch
    std::vector<std::vector<std::vector<std::vector<float>>>> activateBatch;    // for batch

// constructors

    mnn2d() = default;
    mnn2d(int inw, int inh, int outw, int layers, float order, std::string binFileAddress);
    mnn2d(int inw, int inh, int outw, int dim, int layers, float order, std::string binFileAddress);
    mnn2d(int inw, int inh, int outw, std::vector<int> width, float order, std::string binFileAddress);

    void setbatch(int batchsize) { this->batchSize = batchsize; }
    void setepochs(int epochs) { this->epochs = epochs; }
    void setiterations(int iterations) { this->iterations = iterations; }
    void setlearningrate(float learningrate) { this->learningRate = learningrate; }
    void setCWeights(std::vector<std::vector<float>>& cweight, int layerNumber) { cweights[layerNumber - 1] = cweight; }
    void setBWeights(std::vector<std::vector<float>>& bweight, int layerNumber) { bweights[layerNumber - 1] = bweight; }
    void setCGradients(std::vector<std::vector<float>>& cgradient, int layerNumber) { cgradients[layerNumber - 1] = cgradient; }
    void setBGradients(std::vector<std::vector<float>>& bgradient, int layerNumber) { bgradients[layerNumber - 1] = bgradient; }
    void setActivate(std::vector<std::vector<float>>& act, int layerNumber) { activate[layerNumber - 1] = act; }
    void setDotProds(std::vector<std::vector<float>>& dot, int layerNumber) { dotProds[layerNumber - 1] = dot; }
    // access C weight layer (1-based index)
    std::vector<std::vector<float>> getCWeights(int layerNumber) { return cweights[layerNumber-1]; }
    std::vector<std::vector<float>> getBWeights(int layerNumber) { return bweights[layerNumber-1]; }
    std::vector<std::vector<float>> getCGradients(int layerNumber) { return cgradients[layerNumber-1]; }
    std::vector<std::vector<float>> getBGradients(int layerNumber) { return bgradients[layerNumber-1]; }
    std::vector<std::vector<float>> getActivate(int layerNumber) { return activate[layerNumber-1]; }
    std::vector<std::vector<float>> getDotProds(int layerNumber) { return dotProds[layerNumber-1]; }
    // get rows and columns
    int getCWrows(int layerNumber) { return cweights[layerNumber].size(); }
    int getCWcols(int layerNumber) { return cweights[layerNumber][0].size(); }
    int getBWrows(int layerNumber) { return bweights[layerNumber].size(); }
    int getBWcols(int layerNumber) { return bweights[layerNumber][0].size(); }
    int getCGrows(int layerNumber) { return cgradients[layerNumber].size(); }
    int getCGcols(int layerNumber) { return cgradients[layerNumber][0].size(); }
    int getBGrows(int layerNumber) { return bgradients[layerNumber].size(); }
    int getBGcols(int layerNumber) { return bgradients[layerNumber][0].size(); }
    int getArows(int layerNumber) { return activate[layerNumber].size(); }
    int getAcols(int layerNumber) { return activate[layerNumber][0].size(); }
    int getDrows(int layerNumber) { return dotProds[layerNumber].size(); }
    int getDcols(int layerNumber) { return dotProds[layerNumber][0].size(); }

    void makeBinFile(const std::string& fileAddress);

    #ifdef USE_CPU

        void forprop(const std::vector<std::vector<float>>& input);
        void forprop(const std::vector<std::vector<std::vector<float>>>& input);
        void backprop(const std::vector<float>& target);
        void backprop(const std::vector<std::vector<float>>& target);
        void train(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void trainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);
        void threadTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void threadTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);

    #elif USE_CUDA

        void cuForprop(const std::vector<std::vector<float>>& input);
        void cuBackprop(const std::vector<float>& target);
        void cuBackprop(const std::vector<std::vector<float>>& target);
        void cuTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void cuTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);
        void cuBufTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void cuBufTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);

    #elif USE_OPENCL

        cl::Context clContext;               // OpenCL context
        cl::CommandQueue clCommandQueue;     // OpenCL command queue
        std::map<std::string, cl::Kernel> kernels; // Map to store kernel objects by name
        cl_int err;                          // To hold OpenCL error codes

        void clForprop(const std::vector<std::vector<float>>& input);
        void clBackprop(const std::vector<float>& target);
        void clBackprop(const std::vector<std::vector<float>>& target);
        void clTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void clTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);
        void clBufTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void clBufTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);

    #endif

    void train(const std::string& dataSetPath, int batchSize);
    void test(const std::string& dataSetPath, float loss);

// destructor
    ~mnn2d() = default;
};

#endif // MNN_HPP