#ifdef USE_CPU
#include "mnn.hpp"
#include <vector>
#include <stdexcept>

/**
 * @brief Trains the mnn network on a single input-target pair.
 * @param input The input vector.
 * @param target The target output vector.
 */
void mnn::train(const std::vector<float>& input, const std::vector<float>& target) {
    while (1) {
        // 1. Forward propagation
        this->input = input;
        forprop(this->input);

        // check for error and break if acceptable
        float loss = crossEntropy(output, target);

        // 2. Backward propagation
        this->target = target;
        backprop(this->target);

        // 3. Update weights for each layer
        for (int i = 0; i < layers; ++i) {
            updateWeights(cweights[i], cgradients[i], learningRate);
            updateWeights(bweights[i], bgradients[i], learningRate);
        }
    }
}

/**
 * @brief Trains the mnn network on a batch of data.
 * @param inputs A vector of input vectors.
 * @param targets A vector of target vectors.
 */
void mnn::trainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs and targets in batch must be the same.");
    }

    while (1) {
        // Initialize cumulative gradients to zero
        std::vector<std::vector<std::vector<float>>> cumulative_cgradients = cgradients;
        std::vector<std::vector<std::vector<float>>> cumulative_bgradients = bgradients;
        for (auto& layer_grad : cumulative_cgradients) for (auto& row : layer_grad) std::fill(row.begin(), row.end(), 0.0f);
        for (auto& layer_grad : cumulative_bgradients) for (auto& row : layer_grad) std::fill(row.begin(), row.end(), 0.0f);

        // Accumulate gradients over the batch
        for (size_t i = 0; i < inputs.size(); ++i) {
            this->input = inputs[i];
            forprop(this->input);

            this->target = targets[i];
            backprop(this->target);

            // Add current gradients to cumulative gradients
            for (int j = 0; j < layers; ++j) {
                for (size_t k = 0; k < cgradients[j].size(); ++k) {
                    for (size_t l = 0; l < cgradients[j][k].size(); ++l) {
                        cumulative_cgradients[j][k][l] += cgradients[j][k][l];
                        cumulative_bgradients[j][k][l] += bgradients[j][k][l];
                    }
                }
            }
        }

        // Average the gradients and update weights
        float batch_size_float = static_cast<float>(inputs.size());
        for (int i = 0; i < layers; ++i) {
            for (size_t j = 0; j < cgradients[i].size(); ++j) {
                for (size_t k = 0; k < cgradients[i][j].size(); ++k) {
                    cgradients[i][j][k] = cumulative_cgradients[i][j][k] / batch_size_float;
                    bgradients[i][j][k] = cumulative_bgradients[i][j][k] / batch_size_float;
                }
            }
            updateWeights(cweights[i], cgradients[i], learningRate);
            updateWeights(bweights[i], bgradients[i], learningRate);
        }
    }
}

/**
 * @brief Trains the mnn2d network on a single input-target pair.
 * @param input The input matrix.
 * @param target The target vector (corresponding to the pooled output).
 */
void mnn2d::train(const std::vector<std::vector<float>>& input, const std::vector<float>& target) {
    // 1. Forward propagation
    this->input = input;
    forprop(this->input);

    // 2. Backward propagation
    this->target = target;
    backprop(this->target);

    // 3. Update weights for each layer
    for (int i = 0; i < layers; ++i) {
        updateWeights(cweights[i], cgradients[i], learningRate);
        updateWeights(bweights[i], bgradients[i], learningRate);
    }
}

/**
 * @brief Trains the mnn2d network on a batch of data.
 * @param inputs A vector of input matrices.
 * @param targets A vector of target vectors.
 */
void mnn2d::trainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs and targets in batch must be the same.");
    }

    // Initialize cumulative gradients to zero
    std::vector<std::vector<std::vector<float>>> cumulative_cgradients = cgradients;
    std::vector<std::vector<std::vector<float>>> cumulative_bgradients = bgradients;
    for (auto& layer_grad : cumulative_cgradients) for (auto& row : layer_grad) std::fill(row.begin(), row.end(), 0.0f);
    for (auto& layer_grad : cumulative_bgradients) for (auto& row : layer_grad) std::fill(row.begin(), row.end(), 0.0f);

    // Accumulate gradients over the batch
    for (size_t i = 0; i < inputs.size(); ++i) {
        this->input = inputs[i];
        forprop(this->input);

        this->target = targets[i];
        backprop(this->target);

        // Add current gradients to cumulative gradients
        for (int j = 0; j < layers; ++j) {
            for (size_t k = 0; k < cgradients[j].size(); ++k) {
                for (size_t l = 0; l < cgradients[j][k].size(); ++l) {
                    cumulative_cgradients[j][k][l] += cgradients[j][k][l];
                    cumulative_bgradients[j][k][l] += bgradients[j][k][l];
                }
            }
        }
    }

    // Average the gradients and update weights
    float batch_size_float = static_cast<float>(inputs.size());
    for (int i = 0; i < layers; ++i) {
        // Average gradients before updating
        for (size_t j = 0; j < cgradients[i].size(); ++j) {
            for (size_t k = 0; k < cgradients[i][j].size(); ++k) {
                cgradients[i][j][k] = cumulative_cgradients[i][j][k] / batch_size_float;
                bgradients[i][j][k] = cumulative_bgradients[i][j][k] / batch_size_float;
            }
        }
        // Update weights with averaged gradients
        updateWeights(cweights[i], cgradients[i], learningRate);
        updateWeights(bweights[i], bgradients[i], learningRate);
    }
}

#endif