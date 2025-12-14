#ifndef MNN_HPP
#define MNN_HPP 1
#include <vector>
#include <string>
#include <utility>
#include <map>
#include "operators.hpp"
#include "progress.hpp"

/**
 * @brief Class representing a Monomial Neural Network (MNN).
 *      - The monomial is of the form f(x) = c*(x^m) + b
 *          - n: order of monomials
 *          - x: input to monimial`d mlp
 *          - c: coefficient of x^m
 *          - b: constant
 *          - Both c and b are trainable parameters.
 */
class mnn {
public:
    float order;                // order of neurons
    int inSize;                 // input size
    int outSize;                // output size
    int layers;                 // number of hidden layers
    int batchSize;              // number of inputs for batch training
    int epochs;                 // number of epochs
    int iterations;             // number of iterations
    int weightUpdateType;       // type of weight update
    float learningRate;         // learning rate
    float prevloss;             // previous iterations loss
    float currloss;             // current loss
    std::vector<int> width;     // width of each layer and subsequent layers height

    std::vector<float> input;       // input vector
    std::vector<float> output;      // output vector
    std::vector<float> target;      // target vector

    unsigned long long param;       // counter for iterations
    std::string binFileAddress;     // binary file address to save weights and biases
    std::string path2progress;      // path to progress file for session-wise checks
    std::string path2test_progress; // path to test progress file for session-wise checks
    std::string path2EpochDir;      // epoch-based data stored here
    progress trainPrg;              // train progress
    test_progress testPrg;          // test progress
    confMat confData;               // precision and accuracy data
    scores allScores;               // r^2
    std::vector<std::vector<int>> confusion;        // confusion matrix
    std::vector<std::vector<float>> weightStats;    // weight statistics

// store values for training

    std::vector<std::vector<std::vector<float>>> cweights;      // c-coefficients of the network
    std::vector<std::vector<std::vector<float>>> bweights;      // b-constants of the network
    std::vector<std::vector<std::vector<float>>> cgradients;    // gradients of c-coefficients of the network
    std::vector<std::vector<std::vector<float>>> bgradients;    // gradients of b-constants of the network
    std::vector<std::vector<float>> dotProds;       // (activation^n) * cweights + bweights = dot products
    std::vector<std::vector<float>> activate;       // activations of the network

    // for batch-wise training

    std::vector<std::vector<float>> inputBatch;     // input vector
    std::vector<std::vector<float>> outputBatch;    // output vector
    std::vector<std::vector<float>> targetBatch;    // target vector
    std::vector<std::vector<std::vector<float>>> dotBatch; // for batch
    std::vector<std::vector<std::vector<float>>> actBatch; // for batch

// constructors

    mnn() = default;
    mnn(int insize, int outsize, int layers, float order, std::string binFileAddress);
    mnn(int insize, int outsize, int dim, int layers, float order, std::string binFileAddress);
    mnn(int insize, int outsize, std::vector<int> width, float order, std::string binFileAddress);

    void makeBinFile(const std::string& fileAddress);
    void initiateWeights(int type);
    void loadNetwork();
    void saveNetwork();
    friend void serializeWeights(const std::vector<std::vector<std::vector<float>>>& cweights, const std::vector<std::vector<std::vector<float>>>& bweights, 
                                    const std::string& fileAddress);
    friend void deserializeWeights(std::vector<float>& cweights, std::vector<float>& bweights, const std::vector<int>& width, const std::vector<int>& height, 
                                    const std::string& fileAddress);

    #ifdef USE_CPU

        void forprop(const std::vector<float>& input);
        void forprop(const std::vector<std::vector<float>>& input);
        void backprop(const std::vector<float>& target);
        void backprop(const std::vector<std::vector<float>>& target);
        void train1c(const std::vector<float>& input, const std::vector<float>& target, bool useThreadOrBuffer);
        void trainBatch1c(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, bool useThreadOrBuffer);
        void train(const std::vector<float>& input, const std::vector<float>& target);
        void trainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets);
        void threadTrain(const std::vector<float>& input, const std::vector<float>& target);
        void threadTrainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets);

    #elif USE_CU

        void cuForprop(const std::vector<float>& input);
        void cuForprop(const std::vector<std::vector<float>>& input);
        void cuBackprop(const std::vector<float>& target);
        void cuBackprop(const std::vector<std::vector<float>>& target);
        void cuTrain1c(const std::vector<float>& input, const std::vector<float>& target, bool useThreadOrBuffer);
        void cuTrainBatch1c(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, bool useThreadOrBuffer);
        void cuTrain(const std::vector<float>& input, const std::vector<float>& target);
        void cuTrainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets); 
        void cuBufTrain(const std::vector<float>& input, const std::vector<float>& target);
        void cuBufTrainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets);

    #elif USE_CL

        cl::Context clContext;              // OpenCL context
        cl::CommandQueue clCommandQueue;    // OpenCL command queue
        cl::Device device;                  // Represents the selected OpenCL device.
        std::map<std::string, cl::Kernel> kernels;      // Map to store kernel objects by name

        void clForprop(const std::vector<float>& input);
        void clForprop(const std::vector<std::vector<float>>& input);
        void clBackprop(const std::vector<float>& target);
        void clBackprop(const std::vector<std::vector<float>>& target);
        void clTrain1c(const std::vector<float>& input, const std::vector<float>& target, bool useThreadOrBuffer);
        void clTrainBatch1c(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, bool useThreadOrBuffer);
        void clTrain(const std::vector<float>& input, const std::vector<float>& target);
        void clTrainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets);
        void clBufTrain(const std::vector<float>& input, const std::vector<float>& target);
        void clBufTrainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets);

    #endif

    void zeroGradients();
    void onlineTraining(const std::string& dataSetPath, bool useBatchTrain, bool useThreadOrBuffer);
    void miniBatchTraining(const std::string& dataSetPath, bool useThreadOrBuffer);
    void fullDataSetTraining(const std::string& dataSetPath, bool useThreadOrBuffer);
    void test(const std::string& dataSetPath, bool useThreadOrBuffer);

// destructor
    ~mnn() = default;
};

#endif // MNN_HPP