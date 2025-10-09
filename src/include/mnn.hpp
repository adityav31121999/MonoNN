#ifndef MNN_HPP
#define MNN_HPP
#include <vector>
#include <string>
#include "activations.hpp"
#include "loss.hpp"

/**
 * @brief Class representing a Monomial Neural Network (MNN).
        - The monomial is of the form f(x) = c*(x^n) + b
            - x: input to monimial
            - n: order of all monomial, neurons and mlp
            - c: coefficient of x^n
            - b: constant
            - Both c and b are trainable parameters.
 */
class mnn {
private:
    float order;                // order of neurons
    int inSize;                 // input size
    int outSize;                // output size
    std::vector<float> input;      // input vector
    std::vector<float> output;     // output vector
    std::vector<float> target;     // target vector

    int batchSize;               // batch size for training
    std::vector<std::vector<float>> in2d;   // input for batch training
    std::vector<std::vector<float>> out2d;  // output for batch training
    std::vector<std::vector<float>> tar2d;  // target for batch training

    int epochs;                  // number of epochs
    int iterations;              // number of iterations
    float learningRate;          // learning rate
    float lambdaL1;              // L1 regularization parameter
    float lambdaL2;              // L2 regularization parameter
    float decayRate;             // weight decay rate
    float dropoutRate;           // dropout rate

// store values for training

    std::vector<std::vector<std::vector<float>>> cweights;      // c-coefficients of the network
    std::vector<std::vector<std::vector<float>>> bweights;      // b-constants of the network
    std::vector<std::vector<std::vector<float>>> biases;        // biases of the network
    std::vector<std::vector<float>> dotProds;       // activation * layers = dot products
    std::vector<std::vector<float>> activate;       // activations of the network

    std::vector<std::vector<std::vector<float>>> cgradients;    // c-coefficients of the network
    std::vector<std::vector<std::vector<float>>> bgradients;    // b-constants of the network
    std::vector<std::vector<std::vector<float>>> biasgrad;      // biases of the network

    int epochs;                  // number of epochs
    int iterations;              // number of iterations
    float learningRate;          // learning rate
    float lambdaL1;              // L1 regularization parameter
    float lambdaL2;              // L2 regularization parameter
    float decayRate;             // weight decay rate
    float dropoutRate;           // dropout rate

public:
// constructors

    mnn() = default;
    mnn(int insize, int outsize, std::vector<int> width, float order);
    mnn(int insize, int outsize, int dim, float order);

    void forprop();
    void backprop();
    void updateWeights(float learningRate);
    void updateWeightsL1(float learningRate, float lambdaL1);
    void updateWeightsL2(float learningRate, float lambdaL2);
    void updateWeightsElastic(float learningRate, float lambdaL1, float lambdaL2);
    void updateWeightsWeightDecay(float learningRate, float decayRate);
    void updateWeightsDropout(float learning, float dropoutRate);
    void train(std::vector<float> input, std::vector<float> target);
    void trainBatch(std::vector<std::vector<float>> input, std::vector<std::vector<float>> target);

    ~mnn() = default;
};


/**
 * @brief Class representing a Monomial Neural Network (MNN) for 2D i/o.
        - The monomial is of the form f(x) = c*(x^n) + b
            - x: input to monimial
            - n: order of all monomial, neurons and mlp
            - c: coefficient of x^n
            - b: constant
            - Both c and b are trainable parameters.
 */
class mnn2d {
private:
    float order;                    // order of neurons
    int inWidth;                    // input size
    int inHeight;                   // input size
    int outHeight;                  // output size
    int outWidth;                   // output size
    std::vector<int> width;         // width of each layer
    std::vector<int> height;        // height of each layer
    std::vector<std::vector<float>> input;      // input vector
    std::vector<std::vector<float>> output;     // output vector
    std::vector<std::vector<float>> target;     // target vector

    int batchSize;               // batch size for training
    std::vector<std::vector<std::vector<float>>> in2d;      // input for batch training
    std::vector<std::vector<std::vector<float>>> out2d;     // output for batch training
    std::vector<std::vector<std::vector<float>>> tar2d;     // target for batch training

// store values for training

    std::vector<std::vector<std::vector<float>>> cweights;  // c-coefficients of the network
    std::vector<std::vector<std::vector<float>>> bweights;  // b-constants of the network
    std::vector<std::vector<std::vector<float>>> biases;    // biases of the network
    std::vector<std::vector<std::vector<float>>> dotProds;  // activation * layers = dot products
    std::vector<std::vector<std::vector<float>>> activate;  // activations of the network

    std::vector<std::vector<std::vector<float>>> cgradients;    // c-coefficients of the network
    std::vector<std::vector<std::vector<float>>> bgradients;    // b-constants of the network
    std::vector<std::vector<std::vector<float>>> biasgrad;      // biases of the network

public:
// constructors

    mnn2d() = default;
    mnn2d(int inw, int inh, int outw, int outh, std::vector<int> width, float order);
    mnn2d(int inw, int inh, int outw, int outh, int dim, float order);

    void forprop();
    void backprop();
    void updateWeights(float learningRate);
    void updateWeightsL1(float learningRate, float lambdaL1);
    void updateWeightsL2(float learningRate, float lambdaL2);
    void updateWeightsElastic(float learningRate, float lambdaL1, float lambdaL2);
    void updateWeightsWeightDecay(float learningRate, float decayRate);
    void updateWeightsDropout(float learning, float dropoutRate);
    void train(std::vector<std::vector<float>> input, std::vector<std::vector<float>> target);
    void trainBatch(std::vector<std::vector<std::vector<float>>> input, std::vector<std::vector<std::vector<float>>> target);

    ~mnn2d() = default;
};

#endif // MNN_HPP