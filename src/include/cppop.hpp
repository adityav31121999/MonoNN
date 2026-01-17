#ifndef CPPOP_HPP
#define CPPOP_HPP 1

#include <numeric>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

inline float clamp(float val) {
    if (std::isnan(val)) return 0.0f;
    if (std::isinf(val)) return 0.0f;
    return val;
}

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
std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& x);
std::vector<std::vector<float>> softmaxDer(const std::vector<std::vector<float>>& x);
std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& x, float temp);
std::vector<std::vector<float>> softmaxDer(const std::vector<std::vector<float>>& x, float temp);

// errors

float mean(const std::vector<float>&);
float sumOfSquareOfDiff(const std::vector<float>& a, const std::vector<float> b);
float sumOfSquareOfDiff(const std::vector<float>& a, const float b);
float mse(const std::vector<float>& output, const std::vector<float>& target);
float crossEntropy(const std::vector<float>& output, const std::vector<float>& target);
float binaryCrossEntropy(const std::vector<float>& output, const std::vector<float>& target);
float categoricalCrossEntropy(const std::vector<std::vector<float>>& output, const std::vector<std::vector<float>>& target);

// learning rate schedulers

float cosineAnnealing(float MAX_LR, float MIN_LR, int epoch, int totalEpochs);
float learningRateOnPlateau(float currentLR, float previousLoss, float currentLoss, int& patienceCounter, int patience, float factor);

// math operators

std::vector<float> operator+(const std::vector<float>& a, const std::vector<float>& b);
std::vector<std::vector<float>> operator+(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b);
std::vector<float> operator*(const std::vector<float>& a, const std::vector<std::vector<float>>& b);
std::vector<std::vector<float>> operator*(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b);
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

std::vector<std::vector<float>> hadamard(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b);
std::vector<float> multiply(const std::vector<float>& a, const std::vector<float>& b);
std::vector<float> multiply(const std::vector<float>& a, const std::vector<std::vector<float>>& b);
std::vector<std::vector<float>> multiply(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b);
std::vector<float> multiplyWithThreads(const std::vector<float>& a, const std::vector<std::vector<float>>& b);
std::vector<std::vector<float>> multiplyWithThreads(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b);

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

// single layer forprop for mnn

void layerForward(const std::vector<float>& input, std::vector<float>& output, const std::vector<std::vector<float>>& cweights,
                    const std::vector<std::vector<float>>& bweights, float n);
void layerForward(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output, 
                    const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n);

void layerForwardThread(const std::vector<float>& input, std::vector<float>& output, const std::vector<std::vector<float>>& cweights,
                    const std::vector<std::vector<float>>& bweights, float n);
void layerForwardThread(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output, 
                    const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n);

// batch layer forprop for mnn

void layerForwardBatch(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output,
                       const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n);
void layerForwardBatch(const std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<std::vector<float>>>& output,
                       const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n);

void layerForwardBatchThread(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output,
                       const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n);
void layerForwardBatchThread(const std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<std::vector<float>>>& output,
                       const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n);

// single layer backprop (with direct weights update) for mnn and mnn2d for online training

void layerBackward(const std::vector<float>& incoming, const std::vector<float>& input, std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb, float m, float alpha);
void layerBackward(const std::vector<float>& incoming, std::vector<float>& outgoing, const std::vector<float>& prevAct,
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb, float m, float alpha);
void layerBackward(const std::vector<std::vector<float>>& incoming, const std::vector<std::vector<float>>& input,
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb, float m, float alpha);
void layerBackward(const std::vector<std::vector<float>>& incoming, std::vector<std::vector<float>>& outgoing,
                    const std::vector<std::vector<float>>& dotProds, const std::vector<std::vector<float>>& prevAct,
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb, float m, float alpha);

void layerBackwardThread(const std::vector<float>& incoming, const std::vector<float>& input, std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb, float m, float alpha);
void layerBackwardThread(const std::vector<float>& incoming, std::vector<float>& outgoing, const std::vector<float>& prevAct,
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb, float m, float alpha);
void layerBackwardThread(const std::vector<std::vector<float>>& incoming, const std::vector<std::vector<float>>& input,
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb, float m, float alpha);
void layerBackwardThread(const std::vector<std::vector<float>>& incoming, std::vector<std::vector<float>>& outgoing,
                    const std::vector<std::vector<float>>& dotProds, const std::vector<std::vector<float>>& prevAct,
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb, float m, float alpha);

// batch layer backprop (with averaging and direct weights update) for mnn and mnn2d for batch training

void layerBackwardBatch(const std::vector<std::vector<float>>& incoming, const std::vector<std::vector<float>>& input, 
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

void layerBackwardBatchThread(const std::vector<std::vector<float>>& incoming, const std::vector<std::vector<float>>& input, 
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb,
                    float m, float alpha);
void layerBackwardBatchThread(const std::vector<std::vector<std::vector<float>>>& incoming, const std::vector<std::vector<std::vector<float>>>& prevAct,
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb, 
                    float m, float alpha);
void layerBackwardBatchThread(const std::vector<std::vector<float>>& incoming, std::vector<std::vector<float>>& outgoing,
                    const std::vector<std::vector<float>>& prevAct, std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb,
                    float m, float alpha);
void layerBackwardBatchThread(const std::vector<std::vector<std::vector<float>>>& incoming, std::vector<std::vector<std::vector<float>>>& outgoing,
                    const std::vector<std::vector<std::vector<float>>>& dotProds, const std::vector<std::vector<std::vector<float>>>& prevAct,
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb,
                    float m, float alpha);

#endif