#ifdef USE_CPU
#include "mnn.hpp"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>

/**
 * @brief monomial operation for single layer in forprop
 * @param [in] input vector input
 * @param [out] output vector output
 * @param [in] cweights coefficient weights
 * @param [in] bweights bias weights
 */
void layerForward(const std::vector<float>& input, std::vector<float>& output, const std::vector<std::vector<float>>& cweights,
                const std::vector<std::vector<float>>& bweights)
{
    if(input.size() != cweights.size()) {
        throw std::runtime_error("input size and cweights rows mismatch :)");
    }
    if(input.size() != bweights.size()) {
        throw std::runtime_error("input size and bweights rows mismatch :)");
    }
    if(output.size() != cweights[0].size()) {
        throw std::runtime_error("output size and cweights columns mismatch :)");
    }
    if(output.size() != bweights[0].size()) {
        throw std::runtime_error("output size and bweights columns mismatch :)");
    }
    
    for(int i = 0; i < cweights.size(); i++) {
        for(int j = 0; j < cweights[0].size(); j++) {
            // output[j] = sum(input[i]*cweights[i][j] + bweights[i][j]) for j = 0 to height - 1
            output[j] += (input[i]*cweights[i][j]) + bweights[i][j];
        }
    }
    for(int i = 0; i < output.size(); i++) {
        if (std::isnan(output[i])) {
            output[i] = 0.0f;
        }
        else if (std::isinf(output[i])) {
            output[i] = 1.0f;
        }
    }
}


/**
 * @brief monomial operation for single layer in forprop
 * @param [in] input vector input
 * @param [out] output vector output
 * @param [in] cweights coefficient weights
 * @param [in] bweights bias weights
 * @param [in] n order of monomial
 */
void layerForward(const std::vector<float>& input, std::vector<float>& output, const std::vector<std::vector<float>>& cweights,
                const std::vector<std::vector<float>>& bweights, float n)
{
    if(input.size() != cweights.size()) {
        throw std::runtime_error("input size and cweights rows mismatch: " + std::to_string(input.size()) + " != " + std::to_string(cweights.size()));
    }
    if(input.size() != bweights.size()) {
        throw std::runtime_error("input size and bweights rows mismatch: " + std::to_string(input.size()) + " != " + std::to_string(bweights.size()));
    }
    if(output.size() != cweights[0].size()) {
        throw std::runtime_error("output size and cweights columns mismatch: " + std::to_string(output.size()) + " != " + std::to_string(cweights[0].size()));
    }
    if(output.size() != bweights[0].size()) {
        throw std::runtime_error("output size and bweights columns mismatch: " + std::to_string(output.size()) + " != " + std::to_string(bweights[0].size()));
    }
    std::vector<float> powerIn = power(input, n);
    for(int i = 0; i < cweights.size(); i++) {
        for(int j = 0; j < cweights[0].size(); j++) {
            output[j] += (powerIn[i]*cweights[i][j]) + bweights[i][j];
        }
    }
    for(int i = 0; i < output.size(); i++) {
        if (std::isnan(output[i])) {
            output[i] = 0.0f;
        }
        else if (std::isinf(output[i])) {
            output[i] = 1.0f;
        }
    }
}


/**
 * @brief monomial operation for single layer in forprop
 * @param [in] input matrix input
 * @param [out] output matrix output
 * @param [in] cweights coefficient weights
 * @param [in] bweights bias weights
 */
void layerForward(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output,
                    const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights)
{
    if (input.empty() || cweights.empty() || bweights.empty()) {
        throw std::invalid_argument("Input and weight matrices cannot be empty.");
    }
    if (input[0].size() != cweights.size()) {
        throw std::runtime_error("Input columns and cweights rows mismatch.");
    }
    if (cweights.size() != bweights.size() || cweights[0].size() != bweights[0].size()) {
        throw std::runtime_error("cweights and bweights dimensions must match.");
    }
    if (output.size() != input.size() || output[0].size() != cweights[0].size()) {
        throw std::runtime_error("Output matrix has incorrect dimensions.");
    }

    // output = (input^n) * cweights + bweights
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < cweights[0].size(); ++j) {
            for (size_t k = 0; k < cweights.size(); ++k) {
                output[i][j] += (input[i][k] * cweights[k][j]) + bweights[i][j];
            }
            if (std::isnan(output[i][j])) {
                output[i][j] = 0.0f;
            }
            if (std::isinf(output[i][j])) {
                output[i][j] = 1.0f;
            }
        }
    }
}


/**
 * @brief monomial operation for single layer in forprop
 * @param [in] input matrix input
 * @param [out] output matrix output
 * @param [in] cweights coefficient weights
 * @param [in] bweights bias weights
 * @param [in] n order of monomial
 */
void layerForward(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output,
                    const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n)
{
    if (input.empty() || cweights.empty() || bweights.empty()) {
        throw std::invalid_argument("Input and weight matrices cannot be empty.");
    }
    if (input[0].size() != cweights.size()) {
        throw std::runtime_error("Input columns and cweights rows mismatch.");
    }
    if (cweights.size() != bweights.size() || cweights[0].size() != bweights[0].size()) {
        throw std::runtime_error("cweights and bweights dimensions must match.");
    }
    if (output.size() != input.size() || output[0].size() != cweights[0].size()) {
        throw std::runtime_error("Output matrix has incorrect dimensions.");
    }

    // output = (input^n) * cweights + bweights
    std::vector<std::vector<float>> powerIn(input.size(), std::vector<float>(input[0].size(), 0.0f));
    for (size_t i = 0; i < input.size(); ++i) {
        powerIn[i] = power(input[i], n);
    }
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < cweights[0].size(); ++j) {
            float dotProd_ij = 0.0f;
            for (size_t k = 0; k < cweights.size(); ++k) {
                dotProd_ij += (powerIn[i][k] * cweights[k][j]) + bweights[k][j];
            }
            output[i][j] = dotProd_ij; // Assign the final sum
            if (std::isnan(output[i][j])) {
                output[i][j] = 0.0f;
            }
            if (std::isinf(output[i][j])) {
                output[i][j] = 1.0f;
            }
        }
    }
}

#endif