#ifndef MNN2D_HPP
#define MNN2D_HPP 1
#include <vector>
#include <string>
#include <utility>
#include <map>
#include "operators.hpp"
#include "progress.hpp"

/**
 * @brief Class representing a Monomial Neural Network (MNN) for 2D input and 1D output.
 *      - The monomial is of the form f(x) = c*(x^m) + b
 *          - n: order of monomials
 *          - x: input to monimial`d mlp
 *          - c: coefficient of x^m
 *          - b: constant
 *          - Both c and b are trainable parameters.
 */
class mnn2d {
public:
    float order;                    // order of neurons
    int inWidth;                    // input matrix width
    int inHeight;                   // input matrix height
    int outWidth;                   // output matrix width
    int layers;                     // number of hidden layers
    int batchSize;                  // batch size for training
    int epochs;                     // number of epochs
    int iterations;                 // number of iterations
    int weightUpdateType;           // type of weight update
    float learningRate;             // learning rate
    float currloss;                 // current loss
    float prevloss;                 // previous iterations loss
    std::vector<int> width;         // width of each layer

    std::vector<std::vector<float>> input;      // input vector
    std::vector<float> output;      // output vector: Mean Pool activate[layers-1]
    std::vector<float> target;      // target vector

    unsigned long long param;       // counter for iterations
    std::string binFileAddress;     // binary file address to save weights and biases
    std::string path2progress;      // path to progress file
    std::string path2test_progress; // path to test progress file
    progress trainPrg;              // train progress
    test_progress testPrg;          // test progress

// weights and biases

    std::vector<std::vector<std::vector<float>>> cweights;      // c-coefficients of the network
    std::vector<std::vector<std::vector<float>>> bweights;      // b-constants of the network
    std::vector<std::vector<std::vector<float>>> cgradients;    // gradients of c-coefficients of the network
    std::vector<std::vector<std::vector<float>>> bgradients;    // gradients of b-constants of the network
    std::vector<std::vector<std::vector<float>>> dotProds;      // (activation^n) * clayers + blayers = dot products
    std::vector<std::vector<std::vector<float>>> activate;      // activations of the network

    std::vector<std::vector<std::vector<float>>> inputBatch;    // input vector
    std::vector<std::vector<float>> outputBatch;                // output vector
    std::vector<std::vector<float>> targetBatch;                // target vector
    std::vector<std::vector<std::vector<std::vector<float>>>> dotBatch;    // for batch
    std::vector<std::vector<std::vector<std::vector<float>>>> actBatch;    // for batch

// constructors

    mnn2d() = default;
    mnn2d(int inw, int inh, int outw, int layers, float order, std::string binFileAddress);
    mnn2d(int inw, int inh, int outw, int dim, int layers, float order, std::string binFileAddress);
    mnn2d(int inw, int inh, int outw, std::vector<int> width, float order, std::string binFileAddress);

    void makeBinFile(const std::string& fileAddress);
    void initiateWeights(int type);
    void loadNetwork();
    void saveNetwork();
    friend void serializeWeights(const std::vector<std::vector<std::vector<float>>>& cweights, const std::vector<std::vector<std::vector<float>>>& bweights, 
                                    const std::string& fileAddress);
    friend void deserializeWeights(std::vector<float>& cweights, std::vector<float>& bweights, const std::vector<int>& width, const std::vector<int>& height, 
                                    const std::string& fileAddress);

    #ifdef USE_CPU

        void forprop(const std::vector<std::vector<float>>& input);
        void forprop(const std::vector<std::vector<std::vector<float>>>& input);
        void backprop(const std::vector<float>& target);
        void backprop(const std::vector<std::vector<float>>& target);
        void train(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void trainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);
        void threadTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void threadTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);

    #elif USE_CU

        void cuForprop(const std::vector<std::vector<float>>& input);
        void cuForprop(const std::vector<std::vector<std::vector<float>>>& input);
        void cuBackprop(const std::vector<float>& target);
        void cuBackprop(const std::vector<std::vector<float>>& target);
        void cuTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void cuTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);
        void cuBufTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void cuBufTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);

    #elif USE_CL

        cl::Context clContext;              // OpenCL context
        cl::CommandQueue clCommandQueue;    // OpenCL command queue
        cl::Device device;                  // Represents the selected OpenCL device.
        std::map<std::string, cl::Kernel> kernels;      // Map to store kernel objects by name
        cl_int err;                          // To hold OpenCL error codes

        void clForprop(const std::vector<std::vector<float>>& input);
        void clForprop(const std::vector<std::vector<std::vector<float>>>& input);
        void clBackprop(const std::vector<float>& target);
        void clBackprop(const std::vector<std::vector<float>>& target);
        void clTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void clTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);
        void clBufTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target);
        void clBufTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets);

    #endif

    void zeroGradients();
    void onlineTraining(const std::string& dataSetPath, bool useBatchTrain, bool useThreadOrBuffer);
    void miniBatchTraining(const std::string& dataSetPath, bool useThreadOrBuffer);
    void fullDataSetTraining(const std::string& dataSetPath, bool useThreadOrBuffer);
    void test(const std::string& dataSetPath, bool useThreadOrBuffer);

// destructor
    ~mnn2d() = default;
};

#endif // MNN_HPP