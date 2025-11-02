#ifdef USE_CPU
#include "mnn.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>

/**
 * @brief Trains the mnn network on a single input-target pair.
 * @param input The input vector.
 * @param target The target output vector.
 */
void mnn::train(const std::vector<float>& input, const std::vector<float>& target) {
    int i = 0;
    while (1) {
        // 1. Forward propagation
        this->input = input;
        forprop(this->input);

        if(maxIndex(output) != maxIndex(target)) break;
        i++;

        // check for error and break if acceptable
        float loss = crossEntropy(output, target);
        std::cout << "Current CE Loss at epoch " << i << ": " <<loss << std::endl;

        // 2. Backward propagation
        this->target = target;
        backprop(this->target);
    }

    std::cout << "Training complete for this input-target pair." << std::endl;
}

/**
 * @brief Trains the mnn network on a batch of data.
 * @param inputs A vector of input vectors.
 * @param targets A vector of target vectors.
 */
void mnn::trainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets) {
    if (inputs.size() != targets.size() || inputs[0].size() != targets[0].size()) {
        throw std::invalid_argument("Number of inputs and targets in batch must be the same.");
    }
}

/**
 * @brief Trains the mnn2d network on a single input-target pair.
 * @param input The input matrix.
 * @param target The target vector (corresponding to the pooled output).
 */
void mnn2d::train(const std::vector<std::vector<float>>& input, const std::vector<float>& target) {
    int i = 0;
    while (1) {
        // 1. Forward propagation
        this->input = input;
        forprop(this->input);

        if(maxIndex(output) != maxIndex(target)) break;
        i++;

        // check for error and break if acceptable
        float loss = crossEntropy(output, target);
        std::cout << "Current CE Loss at epoch " << i << ": " <<loss << std::endl;

        // 2. Backward propagation
        this->target = target;
        backprop(this->target);
    }

    std::cout << "Training complete for this input-target pair." << std::endl;
}

/**
 * @brief Trains the mnn2d network on a batch of data.
 * @param inputs A vector of input matrices.
 * @param targets A vector of target vectors.
 */
void mnn2d::trainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets) {
    if (inputs.size() != targets.size() || inputs[0].size() != targets[0].size()) {
        throw std::invalid_argument("Number of inputs and targets in batch must be the same.");
    }
}

#endif