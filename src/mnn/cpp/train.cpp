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
    std::cout << "Training on single input-output pair.\n";
    this->target = target;
    int i = 0;
    while (1) {
        // 1. Forward propagation
        this->input = input;
        forprop(this->input);

        // check for error and break if acceptable
        float loss = crossEntropy(output, target);
        std::cout << "Current CE Loss at epoch " << i << ": " <<loss << " with prediction at index: " << maxIndex(output) << std::endl;

        if(maxIndex(output) == maxIndex(target)) {
            std::cout << "Correct output predicted :) at epoch " << i << "." << std::endl;
            break;
        }
        i++;
        if(i == EPOCH) break;

        // 2. Backward propagation
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
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs and targets in batch must be the same.");
    }
    if (inputs.empty()) {
        return; // Nothing to train
    }
    if (inputs[0].size() != inSize || targets[0].size() != outSize) {
        throw std::invalid_argument("Input or target dimensions do not match network configuration.");
    }
 
    this->batchSize = inputs.size();
    int totalEpochs = 0;
    int i = 0;
    while (true) {
        float total_loss = 0.0f;
        forprop(inputs);
        totalEpochs++;
        total_loss = categoricalCrossEntropy(outputBatch, targets);
        std::cout << "Epoch " << totalEpochs << ", Average CE Loss: " << total_loss / inputs.size() << std::endl;
 
        int correct_predictions = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (maxIndex(this->output) == maxIndex(targets[i])) {
                correct_predictions++;
            }
        }

        if (correct_predictions == inputs.size()) {
            std::cout << "All " << inputs.size() << " outputs in the batch are correct after " << totalEpochs << " epochs. Training complete." << std::endl;
            break;
        }
        else {
            std::cout << "predictions: " <<  correct_predictions << "/" << inputs.size() << std::endl;
        }

        i++;
        if(i == EPOCH) break;
        backprop(targets);

        if (totalEpochs == epochs) {
            std::cout << correct_predictions << "/" << inputs.size() << " correct. Increasing epochs by 10 and continuing training." << std::endl;
            this->epochs += 10;
        }
    }
}

/**
 * @brief Trains the mnn2d network on a single input-target pair.
 * @param input The input matrix.
 * @param target The target vector (corresponding to the pooled output).
 */
void mnn2d::train(const std::vector<std::vector<float>>& input, const std::vector<float>& target) {
    std::cout << "Training on single input-output pair.\n";
    this->target = target;
    int i = 0;
    while (1) {
        this->input = input;
        forprop(this->input);

        // check for error and break if acceptable
        float loss = crossEntropy(output, target);
        std::cout << "Current CE Loss at epoch " << i << ": " <<loss << " with prediction at index: " << maxIndex(output) << std::endl;
        if(maxIndex(output) == maxIndex(target)) {
            std::cout << "Correct output predicted :) at epoch " << i << "." << std::endl;
            break;
        }
        i++;
        if(i == EPOCH) break;

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
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs and targets in batch must be the same.");
    }
    if (inputs.empty()) {
        return; // Nothing to train
    }
    if (inputs[0].size() != inHeight || inputs[0][0].size() != inWidth || targets[0].size() != outWidth) {
        throw std::invalid_argument("Input or target dimensions do not match network configuration.");
    }
 
    this->batchSize = inputs.size();
    int totalEpochs = 0;
    int i = 0;
    while (true) {
        float total_loss = 0.0f;
        forprop(inputs);
        totalEpochs++;
        total_loss = categoricalCrossEntropy(outputBatch, targets);
        std::cout << "Epoch " << totalEpochs << ", Average CE Loss: " << total_loss / inputs.size() << std::endl;

        int correct_predictions = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (maxIndex(this->output) == maxIndex(targets[i])) {
                correct_predictions++;
            }
        }

        if (correct_predictions == inputs.size()) {
            std::cout << "All " << inputs.size() << " outputs in the batch are correct after " << totalEpochs << " epochs. Training complete." << std::endl;
            break;
        }
        else {
            std::cout << correct_predictions << "/" << inputs.size() << " correct. Increasing epochs by 10 and continuing training." << std::endl;
            this->epochs += 10;
        }

        i++;
        if(i == EPOCH) break;
        backprop(targets);
    }
}

#endif