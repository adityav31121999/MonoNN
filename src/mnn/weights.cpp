#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include "mnn.hpp"
#include "mnn2d.hpp"

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
                return clamp(w - learningRate * g);
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

// set weights

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
                weight = clamp(dis(gen));
            }
        }
    }
    std::cout << "-> Weights initialized using Normal Distribution (mean=" << mean << ", stddev=" << stddev << ").\n";
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
                weight = clamp(dis(gen));
            }
        }
    }
    std::cout << "-> Weights initialized using Uniform Distribution (lower=" << lower << ", upper=" << upper << ").\n";
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
    std::cout << "-> Weights initialized using Xavier/Glorot initialization with "
              << (uniformOrNot ? "Uniform" : "semi-Uniform") << " distribution.\n";
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
    std::cout << "-> Weights initialized using He initialization.\n";
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
    std::cout << "-> Weights initialized using LeCun initialization.\n";
}


/**
 * @brief mnn: intialise weights based on different type
 * @param type for deciding what to use for initialisation:
 *      -> 0: normal distribution based on user defined mean and standard deviation
 *      -> 1: uniform distribution based on user defined upper and lower bound
 *      -> 2: xavier-glorot initialisation
 *      -> 3: He initialisation
 *      -> 4: LeCunn initialisation
 */
void mnn::initiateWeights(int type)
{
    switch (type)
    {
        case 0: {
            float mean = 0.1f;
            float stddev = 0.1f;
            std::cout << "Provide mean and standard deviation for distribution:";
            std::cin >> mean;
            std::cin >> stddev;
            setWeightsByNormalDist(cweights, mean, stddev);
            setWeightsByNormalDist(bweights, mean, stddev);
            break;
        }
        case 1: {
            float lower = -1.0f;
            float upper = 1.0f;
            std::cout << "Provide lower and upper bounds for distribution:";
            std::cin >> lower;
            std::cin >> upper;
            setWeightsByUniformDist(cweights, lower, upper);
            setWeightsByUniformDist(bweights, lower, upper);
            break;
        }
        case 2: {
            std::cout << "Set boolean for uniform or not: ";
            bool uni;
            std::cin >> uni;
            setWeightsByXavier(cweights, inSize, outSize, uni);
            setWeightsByXavier(bweights, inSize, outSize, uni);
            break;
        }
        case 3: {
            setWeightsByHe(cweights, inSize, outSize);
            setWeightsByHe(bweights, inSize, outSize);
            break;
        }
        case 4: {
            setWeightsByLeCunn(cweights, inSize, outSize);
            setWeightsByLeCunn(bweights, inSize, outSize);
            break;
        }
        default: {
            std::cout << "Invalid type" << std::endl;
            break;
        }
    }
    // serializeWeights(cweights, bweights, binFileAddress);
}

/**
 * @brief mnn2d: intialise weights based on different type
 * @param type for deciding what to use for initialisation:
 *      -> 0: normal distribution based on user defined mean and standard deviation
 *      -> 1: uniform distribution based on user defined upper and lower bound
 *      -> 2: xavier-glorot initialisation
 *      -> 3: He initialisation
 *      -> 4: LeCunn initialisation
 */
void mnn2d::initiateWeights(int type)
{
    switch (type)
    {
        case 0: {
            std::cout << "Set weights using a normal (Gaussian) distribution." << std::endl;
            float mean = 0.1f;
            float stddev = 0.1f;
            std::cout << "Provide mean and standard deviation for distribution:";
            std::cin >> mean;
            std::cin >> stddev;
            setWeightsByNormalDist(cweights, mean, stddev);
            setWeightsByNormalDist(bweights, mean, stddev);
            break;
        }
        case 1: {
            std::cout << "Set weights using a uniform distribution." << std::endl;
            float lower = -1.0f;
            float upper = 1.0f;
            std::cout << "Provide lower and upper bounds for distribution:";
            std::cin >> lower;
            std::cin >> upper;
            setWeightsByUniformDist(cweights, lower, upper);
            setWeightsByUniformDist(bweights, lower, upper);
            break;
        }
        case 2: {
            std::cout << "Set weights using Xavier/Glorot initialization." << std::endl;
            std::cout << "Set boolean for uniform or not: ";
            bool uni;
            std::cin >> uni;
            setWeightsByXavier(cweights, inHeight * inWidth, outWidth, uni);
            setWeightsByXavier(bweights, inHeight * inWidth, outWidth, uni);
            break;
        }
        case 3: {
            std::cout << "Set weights using He initialization." << std::endl;
            setWeightsByHe(cweights, inHeight * inWidth, outWidth);
            setWeightsByHe(bweights, inHeight * inWidth, outWidth);
            break;
        }
        case 4: {
            std::cout << "Set weights using LeCun initialization." << std::endl;
            setWeightsByLeCunn(cweights, inHeight * inWidth, outWidth);
            setWeightsByLeCunn(bweights, inHeight * inWidth, outWidth);
            break;
        }
        default: {
            std::cout << "Invalid type" << std::endl;
            break;
        }
    }
    // serializeWeights(cweights, bweights, binFileAddress);
}