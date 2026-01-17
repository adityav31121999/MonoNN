#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "mnn1d.hpp"
#include "mnn2d.hpp"

// zero out gradients for new backprop
void mnn1d::zeroGradients() {
    for(size_t i = 0; i < layers; ++i) {
        if (!cweights[i].empty()) {
            cgradients[i].assign(cweights[i].size(), std::vector<float>(cweights[i][0].size(), 0.0f));
        }
        if (!bweights[i].empty()) {
            bgradients[i].assign(bweights[i].size(), std::vector<float>(bweights[i][0].size(), 0.0f));
        }
    }
}


// zero out gradients for new backprop
void mnn2d::zeroGradients() {
    for(size_t i = 0; i < layers; ++i) {
        if (!cweights[i].empty()) {
            cgradients[i].assign(cweights[i].size(), std::vector<float>(cweights[i][0].size(), 0.0f));
        }
        if (!bweights[i].empty()) {
            bgradients[i].assign(bweights[i].size(), std::vector<float>(bweights[i][0].size(), 0.0f));
        }
    }
}

// update weights

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
                return clamp(w - learningRate * g);
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
                return clamp(w - learningRate * (g + l1_grad));
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
                return clamp(w - learningRate * (g + 2.0f * lambdaL2 * w));
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
                return clamp(w - learningRate * (g + l1_grad + 2.0f * lambdaL2 * w));
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
                return clamp(w * (1.0f - decayRate) - learningRate * g);
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
                // dropout applied, update with elastic net regularization
                // return clamp(w - (learningRate * g+ std::copysignf(LAMBDA_L1, w) + (2.0f * LAMBDA_L2 * w)));
                return clamp(w - (learningRate * g));
            });
    }
}

/**
 * @brief update wieights using gradients using given type
 * @param weights original value matrix
 * @param gradient gradients for values
 * @param learningRate step size of change to be applied
 * @param type for update method (0-5) simple (0), L1(1),
 *      L2(2), elastic net(3), weight decay(4), dropout(5).
 */
void updateWeights(std::vector<std::vector<float>> &weights, std::vector<std::vector<float>> &gradients, float &learningRate, int type)
{
    // update weights
    switch (type) {
        case 0:
            // updateWeights
            updateWeights(weights, gradients, learningRate);
            break;
        case 1:
            // updateWeightsL1
            updateWeightsL1(weights, gradients, learningRate, LAMBDA_L1);
            break;
        case 2:
            // updateWeightsL2
            updateWeightsL2(weights, gradients, learningRate, LAMBDA_L2);
            break;
        case 3:
            // updateWeightsElasticNet
            updateWeightsElastic(weights, gradients, learningRate, LAMBDA_L1, LAMBDA_L2);
            break;
        case 4:
            // updateWeightsWeightDecay
            updateWeightsWeightDecay(weights, gradients, learningRate, WEIGHT_DECAY);
            break;
        case 5:
            // updateWeightsDropout
            updateWeightsDropout(weights, gradients, learningRate, DROPOUT_RATE);
            break;
        default:
            std::cout << "Invalid update type" << std::endl;
            break;
    }
}

void mnn1d::getLayerVariance(const std::string& csvad)
{
    std::vector<float> c(param/2, 0.0f);
    std::vector<float> b(param/2, 0.0f);
    deserializeWeights(c, b, initialValues);
    
    std::vector<std::vector<std::vector<float>>> orw(cweights.size());
    std::vector<std::vector<std::vector<float>>> orb(bweights.size());
    unsigned long long offset = 0;

    for(size_t i = 0; i < cweights.size(); i++) {
        size_t rows = cweights[i].size();
        size_t cols = cweights[i].empty() ? 0 : cweights[i][0].size();

        orw[i].resize(rows, std::vector<float>(cols));
        orb[i].resize(rows, std::vector<float>(cols));

        for(size_t j = 0; j < rows; j++) {
            for(size_t k = 0; k < cols; k++) {
                orw[i][j][k] = c[offset + j * cols + k];
                orb[i][j][k] = b[offset + j * cols + k];
            }
        }
        offset += rows * cols;
    }
    std::cout << "Binary File " << initialValues << " loaded successfully." << std::endl;
    computeLayerVariance(orw, orb, cweights, bweights, csvad);
}

void mnn2d::getLayerVariance(const std::string& stats)
{
    std::vector<float> c(param/2, 0.0f);
    std::vector<float> b(param/2, 0.0f);
    deserializeWeights(c, b, initialValues);
    
    std::vector<std::vector<std::vector<float>>> orw(cweights.size());
    std::vector<std::vector<std::vector<float>>> orb(bweights.size());
    unsigned long long offset = 0;

    for(size_t i = 0; i < cweights.size(); i++) {
        size_t rows = cweights[i].size();
        size_t cols = cweights[i].empty() ? 0 : cweights[i][0].size();

        orw[i].resize(rows, std::vector<float>(cols));
        orb[i].resize(rows, std::vector<float>(cols));

        for(size_t j = 0; j < rows; j++) {
            for(size_t k = 0; k < cols; k++) {
                orw[i][j][k] = c[offset + j * cols + k];
                orb[i][j][k] = b[offset + j * cols + k];
            }
        }
        offset += rows * cols;
    }
    std::cout << "Binary File " << initialValues << " loaded successfully." << std::endl;
    computeLayerVariance(orw, orb, cweights, bweights, stats);
}