#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <filesystem>
#include "mnn.hpp"
#include "mnn2d.hpp"
#include "operators.hpp"
// set weights

/**
 * @brief Initialize weights using a normal (Gaussian) distribution.
 * @param weights The 3D vector of weights for the entire network to be initialized. Passed by reference.
 * @param mean The mean of the normal distribution.
 * @param stddev The standard deviation of the normal distribution.
 */
void setWeightsByNormalDist(std::vector<std::vector<float>>& weights, float mean, float stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(mean, stddev);
    for (auto& row : weights) {
        for (auto& weight : row) {
            weight = clamp(dis(gen));
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
void setWeightsByUniformDist(std::vector<std::vector<float>>& weights, float lower, float upper) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(lower, upper);
    for (auto& row : weights) {
        for (auto& weight : row) {
            weight = clamp(dis(gen));
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
void setWeightsByXavier(std::vector<std::vector<float>>& weights, int fin, int fout, bool uniformOrNot) {
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
void setWeightsByHe(std::vector<std::vector<float>>& weights, int fin, int fout) {
    float stddev = std::sqrt(2.0f / fin);
    setWeightsByNormalDist(weights, 0.01f, stddev);
}

/**
 * @brief Initialize weights using LeCun initialization.
 * @param weights The 3D vector of weights for the entire network to be initialized. Passed by reference.
 * @param fin The number of input units in the weight tensor.
 * @param fout The number of output units in the weight tensor (unused in this implementation).
 */
void setWeightsByLeCunn(std::vector<std::vector<float>>& weights, int fin, int fout) {
    float stddev = std::sqrt(1.0f / fin);
    setWeightsByNormalDist(weights, 0.0f, stddev);
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
    setWeightsByNormalDist(weights, 0.01f, stddev);
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
}


/**
 * @brief mnn: intialise weights based on different type
 * @param type for deciding what to use for initialisation:
 *      -> 0: normal distribution based on user defined mean and standard deviation
 *      -> 1: uniform distribution based on user defined upper and lower bound
 *      -> 2: xavier-glorot initialisation
 *      -> 3: He initialisation
 *      -> 4: LeCunn initialisation
 * @param mixedOrLayered inititalise weights mixed (1) or layer-wise (0)
 */
void mnn::initiateWeights(int type, bool mixedOrLayered)
{
    switch (type)
    {
        case 0: {
            float mean = 0.1f;
            float stddev = 0.1f;
            std::cout << "Provide mean and standard deviation for distribution:";
            std::cin >> mean;
            std::cin >> stddev;
            if (mixedOrLayered == 1) {
                setWeightsByNormalDist(cweights, mean, stddev);
                setWeightsByNormalDist(bweights, mean, stddev);
            }
            else {
                for (int i = 0; i < layers; i++) {
                    setWeightsByNormalDist(cweights[i], mean, stddev);
                    setWeightsByNormalDist(bweights[i], mean/10, stddev);
                }
            }
            break;
        }
        case 1: {
            float lower = -1.0f;
            float upper = 1.0f;
            std::cout << "Provide lower and upper bounds for distribution:";
            std::cin >> lower;
            std::cin >> upper;
            if (mixedOrLayered == 1) {
                setWeightsByUniformDist(cweights, lower, upper);
                setWeightsByUniformDist(bweights, lower, upper);
            }
            else {
                for (int i = 0; i < layers; i++) {
                    setWeightsByUniformDist(cweights[i], lower, upper);
                    setWeightsByUniformDist(bweights[i], lower/10, upper/10);
                }
            }
            break;
        }
        case 2: {
            std::cout << "Set boolean for uniform or not: ";
            bool uni;
            std::cin >> uni;
            if (mixedOrLayered == 1) {
                setWeightsByXavier(cweights, inSize, outSize, uni);
                setWeightsByXavier(bweights, inSize, outSize, uni);
            }
            else {
                for (int i = 0; i < layers; i++) {
                    setWeightsByXavier(cweights[i], cweights[i].size(), cweights[i][0].size(), uni);
                    setWeightsByXavier(bweights[i], cweights[i].size()/10, cweights[i][0].size()/10, uni);
                }
            }
            break;
        }
        case 3: {
            std::cout << "-> Weights initialized using He initialization.\n";
            if (mixedOrLayered == 1) {
                std::cout << "C: "; setWeightsByHe(cweights, inSize, outSize);
                std::cout << "B: "; setWeightsByHe(bweights, inSize, outSize);
            }
            else {
                for (int i = 0; i < layers; i++) {
                    std::cout << "C[" << i << "]: "; setWeightsByHe(cweights[i], cweights[i].size(), cweights[i][0].size());
                    std::cout << "B[" << i << "]: "; setWeightsByHe(bweights[i], cweights[i].size()/10, cweights[i][0].size()/10);
                }
            }
            break;
        }
        case 4: {
            std::cout << "-> Weights initialized using LeCunn initialization.\n";
            if (mixedOrLayered == 1) {
                setWeightsByLeCunn(cweights, inSize, outSize);
                setWeightsByLeCunn(bweights, inSize, outSize);
            }
            else {
                for (int i = 0; i < layers; i++) {
                    setWeightsByLeCunn(cweights[i], cweights[i].size(), cweights[i][0].size());
                    setWeightsByLeCunn(bweights[i], cweights[i].size()/10, cweights[i][0].size()/10);
                }
            }
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
 * @param mixedOrLayered inititalise weights alltogether or layer-wise
 */
void mnn2d::initiateWeights(int type, bool mixedOrLayered)
{
    switch (type)
    {
        case 0: {
            float mean = 0.1f;
            float stddev = 0.1f;
            std::cout << "Provide mean and standard deviation for distribution:";
            std::cin >> mean;
            std::cin >> stddev;
            if (mixedOrLayered == 1) {
                setWeightsByNormalDist(cweights, mean, stddev);
                setWeightsByNormalDist(bweights, mean, stddev);
            }
            else {
                for (int i = 0; i < layers; i++) {
                    setWeightsByNormalDist(cweights[i], mean, stddev);
                    setWeightsByNormalDist(bweights[i], mean/10, stddev);
                }
            }
            break;
        }
        case 1: {
            float lower = -1.0f;
            float upper = 1.0f;
            std::cout << "Provide lower and upper bounds for distribution:";
            std::cin >> lower;
            std::cin >> upper;
            if (mixedOrLayered == 1) {
                setWeightsByUniformDist(cweights, lower, upper);
                setWeightsByUniformDist(bweights, lower, upper);
            }
            else {
                for (int i = 0; i < layers; i++) {
                    setWeightsByUniformDist(cweights[i], lower, upper);
                    setWeightsByUniformDist(bweights[i], lower/10, upper/10);
                }
            }
            break;
        }
        case 2: {
            std::cout << "Set boolean for uniform or not: ";
            bool uni;
            std::cin >> uni;
            if (mixedOrLayered == 1) {
                setWeightsByXavier(cweights, inHeight * inWidth, outSize, uni);
                setWeightsByXavier(bweights, inHeight * inWidth, outSize, uni);
            }
            else {
                for (int i = 0; i < layers; i++) {
                    setWeightsByXavier(cweights[i], cweights[i].size(), cweights[i][0].size(), uni);
                    setWeightsByXavier(bweights[i], cweights[i].size()/10, cweights[i][0].size()/10, uni);
                }
            }
            break;
        }
        case 3: {
            std::cout << "-> Weights initialized using He initialization.\n";
            if (mixedOrLayered == 1) {
                std::cout << "C: "; setWeightsByHe(cweights, inHeight * inWidth, outSize);
                std::cout << "B: "; setWeightsByHe(bweights, inHeight * inWidth, outSize);
            }
            else {
                for (int i = 0; i < layers; i++) {
                    std::cout << "C[" << i << "]: "; setWeightsByHe(cweights[i], cweights[i].size(), cweights[i][0].size());
                    std::cout << "B[" << i << "]: "; setWeightsByHe(bweights[i], cweights[i].size()/10, cweights[i][0].size()/10);
                }
            }
            break;
        }
        case 4: {
            std::cout << "-> Weights initialized using LeCunn initialization.\n";
            if (mixedOrLayered == 1) {
                setWeightsByLeCunn(cweights, inHeight * inWidth, outSize);
                setWeightsByLeCunn(bweights, inHeight * inWidth, outSize);
            }
            else {
                for (int i = 0; i < layers; i++) {
                    setWeightsByLeCunn(cweights[i], cweights[i].size(), cweights[i][0].size());
                    setWeightsByLeCunn(bweights[i], cweights[i].size()/10, cweights[i][0].size()/10);
                }
            }
            break;
        }
        default: {
            std::cout << "Invalid type" << std::endl;
            break;
        }
    }
    // serializeWeights(cweights, bweights, binFileAddress);
}