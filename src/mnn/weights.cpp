#include "mnn.hpp"
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

/**
 * @brief Update weights using standard gradient descent.
 * @param weights The weights of a layer to be updated. Passed by reference.
 * @param gradients The gradients corresponding to the weights.
 * @param learningRate The learning rate for the update step.
 */
void updateWeights(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float& learningRate) {
    for (size_t i = 0; i < weights.size(); ++i) {
        std::transform(weights[i].begin(), weights[i].end(), gradients[i].begin(), weights[i].begin(),
            [learningRate](float w, float g) { 
                return w - learningRate * g; 
            }
        );
    }
}

/**
 * @brief Update weights using gradient descent with L1 regularization.
 * @param weights The weights of a layer to be updated. Passed by reference.
 * @param gradients The gradients corresponding to the weights.
 * @param learningRate The learning rate for the update step.
 * @param lambdaL1 The L1 regularization parameter.
 */
void updateWeightsL1(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learningRate, float lambdaL1) {
    for (size_t i = 0; i < weights.size(); ++i) {
        std::transform(weights[i].begin(), weights[i].end(), gradients[i].begin(), weights[i].begin(),
            [learningRate, lambdaL1](float w, float g) {
                float l1_grad = (w > 0.0f) ? lambdaL1 : ((w < 0.0f) ? -lambdaL1 : 0.0f);
                return w - learningRate * (g + l1_grad);
            }
        );
    }
}

/**
 * @brief Update weights using gradient descent with L2 regularization.
 * @param weights The weights of a layer to be updated. Passed by reference.
 * @param gradients The gradients corresponding to the weights.
 * @param learningRate The learning rate for the update step.
 * @param lambdaL2 The L2 regularization parameter.
 */
void updateWeightsL2(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learningRate, float lambdaL2) {
    for (size_t i = 0; i < weights.size(); ++i) {
        std::transform(weights[i].begin(), weights[i].end(), gradients[i].begin(), weights[i].begin(),
            [learningRate, lambdaL2](float w, float g) {
                float l2_grad = 2.0f * lambdaL2 * w;
                return w - learningRate * (g + l2_grad);
            });
    }
}

/**
 * @brief Update weights using gradient descent with Elastic Net (L1 and L2) regularization.
 * @param weights The weights of a layer to be updated. Passed by reference.
 * @param gradients The gradients corresponding to the weights.
 * @param learningRate The learning rate for the update step.
 * @param lambdaL1 The L1 regularization parameter.
 * @param lambdaL2 The L2 regularization parameter.
 */
void updateWeightsElastic(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learningRate, float lambdaL1, float lambdaL2) {
    for (size_t i = 0; i < weights.size(); ++i) {
        std::transform(weights[i].begin(), weights[i].end(), gradients[i].begin(), weights[i].begin(),
            [learningRate, lambdaL1, lambdaL2](float w, float g) {
                float l1_grad = (w > 0.0f) ? lambdaL1 : ((w < 0.0f) ? -lambdaL1 : 0.0f);
                float l2_grad = 2.0f * lambdaL2 * w;
                return w - learningRate * (g + l1_grad + l2_grad);
            });
    }
}

/**
 * @brief Update weights using gradient descent with weight decay.
 * @param weights The weights of a layer to be updated. Passed by reference.
 * @param gradients The gradients corresponding to the weights.
 * @param learningRate The learning rate for the update step.
 * @param decayRate The rate at which weights decay.
 */
void updateWeightsWeightDecay(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learningRate, float decayRate) {
    for (size_t i = 0; i < weights.size(); ++i) {
        std::transform(weights[i].begin(), weights[i].end(), gradients[i].begin(), weights[i].begin(),
            [learningRate, decayRate](float w, float g) {
                return w * (1.0f - decayRate) - learningRate * g;
            });
    }
}

/**
 * @brief Update weights with a dropout mechanism.
 * @details Some weights are randomly not updated based on the dropout rate. This is a form of regularization applied during training.
 * @param weights The weights of a layer to be updated. Passed by reference.
 * @param gradients The gradients corresponding to the weights.
 * @param learningRate The learning rate for the update step.
 * @param dropoutRate The probability of a weight update being "dropped out" (skipped).
 */
void updateWeightsDropout(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& gradients, float learningRate, float dropoutRate) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < weights.size(); ++i) {
        std::transform(weights[i].begin(), weights[i].end(), gradients[i].begin(), weights[i].begin(),
            [&](float w, float g) {
                if (dis(gen) < dropoutRate) {
                    return w; // Keep weight unchanged (gradient is effectively zero)
                }
                return w - learningRate * g;
            });
    }
}

/**
 * @brief Initialize weights using a normal (Gaussian) distribution.
 * @param weights The 3D vector of weights for the entire network to be initialized. Passed by reference.
 * @param mean The mean of the normal distribution.
 * @param stddev The standard deviation of the normal distribution.
 */
void setWeightsByNormalDist(std::vector<std::vector<std::vector<float>>>& weights, float mean, float stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(mean, stddev);
    for (auto& layer : weights) {
        for (auto& row : layer) {
            for (auto& weight : row) {
                weight = dis(gen);
            }
        }
    }
    std::cout << "Weights initialized using Normal Distribution (mean=" << mean << ", stddev=" << stddev << ").\n";
}

/**
 * @brief Initialize weights using a uniform distribution.
 * @param weights The 3D vector of weights for the entire network to be initialized. Passed by reference.
 * @param lower The lower bound of the uniform distribution.
 * @param upper The upper bound of the uniform distribution.
 */
void setWeightsByUniformDist(std::vector<std::vector<std::vector<float>>>& weights, float lower, float upper) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(lower, upper);
    for (auto& layer : weights) {
        for (auto& row : layer) {
            for (auto& weight : row) {
                weight = dis(gen);
            }
        }
    }
    std::cout << "Weights initialized using Uniform Distribution (lower=" << lower << ", upper=" << upper << ").\n";
}

/**
 * @brief Initialize weights using Xavier/Glorot initialization.
 * @details Can use either a uniform or normal distribution.
 * @param weights The 3D vector of weights for the entire network to be initialized. Passed by reference.
 * @param fin The number of input units in the weight tensor.
 * @param fout The number of output units in the weight tensor.
 * @param uniformOrNot If true, use uniform distribution. If false, this implementation defaults to a uniform 
 *  distribution with a modified range.
 */
void setWeightsByXavier(std::vector<std::vector<std::vector<float>>>& weights, int fin, int fout, bool uniformOrNot) {
    float limit = std::sqrt(6.0f / (fin + fout));
    setWeightsByUniformDist(weights, (uniformOrNot == 1 ? -limit : 0), limit);
    std::cout << "Weights initialized using Xavier/Glorot initialization with " << (uniformOrNot ? "Uniform" : "Normal") << " distribution.\n";
}

/**
 * @brief Initialize weights using He initialization.
 * @details Best for layers with ReLU activation.
 * @param weights The 3D vector of weights for the entire network to be initialized. Passed by reference.
 * @param fin The number of input units in the weight tensor.
 * @param fout The number of output units in the weight tensor (unused in this implementation).
 */
void setWeightsByHe(std::vector<std::vector<std::vector<float>>>& weights, int fin, int fout) {
    float stddev = std::sqrt(2.0f / fin);
    setWeightsByNormalDist(weights, 0.0f, stddev);
    std::cout << "Weights initialized using He initialization.\n";
}

/**
 * @brief Initialize weights using LeCun initialization.
 * @param weights The 3D vector of weights for the entire network to be initialized. Passed by reference.
 * @param fin The number of input units in the weight tensor.
 * @param fout The number of output units in the weight tensor (unused in this implementation).
 */
void setWeightsByLeCunn(std::vector<std::vector<std::vector<float>>>& weights, int fin, int fout) {
    float stddev = std::sqrt(1.0f / fin);
    setWeightsByNormalDist(weights, 0.0f, stddev);
    std::cout << "Weights initialized using LeCun initialization.\n";
}
