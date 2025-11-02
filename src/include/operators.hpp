#ifndef LOSS_HPP
#define LOSS_HPP 1
#include <vector>
#include <cmath>

float mse(const std::vector<float>& output, const std::vector<float>& target);
float crossEntropy(const std::vector<float>& output, const std::vector<float>& target);
float binaryCrossEntropy(const std::vector<float>& output, const std::vector<float>& target);

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
std::vector<float> weightedMeanPool(const std::vector<float>& weights, const std::vector<std::vector<float>>& input);
std::vector<float> flatten(const std::vector<std::vector<float>>& input);
std::vector<std::vector<float>> reshape(const std::vector<float>& input, int rows, int cols);
std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& input);
int maxIndex(const std::vector<float>& input);

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

// single layer forprop

void layerForward(const std::vector<float>& input, std::vector<float>& output, const std::vector<std::vector<float>>& cweights,
                    const std::vector<std::vector<float>>& bweights);
void layerForward(const std::vector<float>& input, std::vector<float>& output, const std::vector<std::vector<float>>& cweights,
                    const std::vector<std::vector<float>>& bweights, float n);
void layerForward(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output, 
                    const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights);
void layerForward(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output, 
                    const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n);

// single layer backprop

void layerBackward(const std::vector<float>& incoming, const std::vector<float>& prevAct, 
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& B, 
                    std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb, 
                    float m, float alpha, float learning, int typeOfUpdate);
void layerBackward(const std::vector<float>& incoming, std::vector<float>& outgoing, const std::vector<float>& prevAct, 
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& B, 
                    std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb,
                    float m, float alpha, float learning, int typeOfUpdate);

void layerBackward(const std::vector<std::vector<float>>& incoming, const std::vector<std::vector<float>>& dotProds, const std::vector<std::vector<float>>& prevAct,
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& B, 
                    std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb, 
                    float m, float alpha, float learning, int typeOfUpdate);
void layerBackward(const std::vector<std::vector<float>>& incoming, std::vector<std::vector<float>>& outgoing,
                    const std::vector<std::vector<float>>& dotProds, const std::vector<std::vector<float>>& prevAct,
                    std::vector<std::vector<float>>& C, std::vector<std::vector<float>>& B,
                    std::vector<std::vector<float>>& gradc, std::vector<std::vector<float>>& gradb,
                    float m, float alpha, float learning, int typeOfUpdate);

#endif // LOSS_HPP