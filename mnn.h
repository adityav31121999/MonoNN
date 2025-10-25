#ifndef MONONN_H
#define MONONN_H

#include <mononn.hpp>

#endif

/**

std::vector<std::vector<float>> updateWeights(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float& learningRate);
std::vector<std::vector<float>> updateWeightsL1(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learningRate, float lambdaL1);
std::vector<std::vector<float>> updateWeightsL2(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learningRate, float lambdaL2);
std::vector<std::vector<float>> updateWeightsElastic(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learningRate, float lambdaL1, float lambdaL2);
std::vector<std::vector<float>> updateWeightsWeightDecay(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learningRate, float decayRate);
std::vector<std::vector<float>> updateWeightsDropout(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learning, float dropoutRate);
std::vector<std::vector<std::vector<float>>> setWeightsByNormalDist(std::vector<std::vector<std::vector<float>>>& weights, float mean, float stddev);
std::vector<std::vector<std::vector<float>>> setWeightsByUniformDist(std::vector<std::vector<std::vector<float>>>& weights, float lower, float upper);
std::vector<std::vector<std::vector<float>>> setWeightsByXavier(std::vector<std::vector<std::vector<float>>>& weights, int fin, int fout);
std::vector<std::vector<std::vector<float>>> setWeightsByHe(std::vector<std::vector<std::vector<float>>>& weights, int fin, int fout);
std::vector<std::vector<std::vector<float>>> setWeightsByLeCunn(std::vector<std::vector<std::vector<float>>>& weights, int fin, int fout);
std::vector<std::vector<std::vector<float>>> setWeightsByNormalDist(int r, int c, int n, float mean, float stddev);
std::vector<std::vector<std::vector<float>>> setWeightsByUniformDist(int r, int c, int n, float lower, float upper);
std::vector<std::vector<std::vector<float>>> setWeightsByXavier(int r, int c, int n, int fin, int fout);
std::vector<std::vector<std::vector<float>>> setWeightsByHe(int r, int c, int n, int fin, int fout);
std::vector<std::vector<std::vector<float>>> setWeightsByLeCunn(int r, int c, int n, int fin, int fout);

 */