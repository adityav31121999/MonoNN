#ifdef USE_CL
#include "mnn.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>


/**
 * @brief clTrains the mnn network on a single input-target pair.
 * @param input The input vector.
 * @param target The target output vector.
 */
void mnn::clTrain(const std::vector<float>& input, const std::vector<float>& target) {
    int i = 0;
    while (1) {
        // 1. Forward propagation
        this->input = input;
        clForprop(this->input);

        if(maxIndex(output) != maxIndex(target)) {
            std::cout << "Correct output predicted :) at epoch " << i << "." << std::endl;
            break;
        }
        i++;

        // check for error and break if acceptable
        float loss = crossEntropy(output, target);
        std::cout << "Current CE Loss at epoch " << i << ": " <<loss << std::endl;
        if (i == EPOCH) break;
        // 2. Backward propagation
        this->target = target;
        clBackprop(this->target);
    }

    std::cout << "clTraining complete for this input-target pair." << std::endl;
}

/**
 * @brief clTrains the mnn network on a batch of data.
 * @param inputs A vector of input vectors.
 * @param targets A vector of target vectors.
 */
void mnn::clTrainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs and targets in batch must be the same.");
    }
    if (inputs.empty()) {
        return; // Nothing to clTrain
    }
    if (inputs[0].size() != inSize || targets[0].size() != outSize) {
        throw std::invalid_argument("Input or target dimensions do not match network configuration.");
    }
 
    this->batchSize = inputs.size();
    int totalEpochs = 0;
 
    while (true) {
        for (int e = 0; e < this->epochs; ++e) {
            float total_loss = 0.0f;
            for (size_t i = 0; i < inputs.size(); ++i) {
                this->input = inputs[i];
                clForprop(this->input);
                total_loss += crossEntropy(this->output, targets[i]);
            }
            totalEpochs++;
            std::cout << "Epoch " << totalEpochs << ", Average CE Loss: " << total_loss / inputs.size() << std::endl;
            clBackprop(const_cast<std::vector<std::vector<float>>&>(targets));
        }
 
        int correct_predictions = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            clForprop(inputs[i]);
            if (maxIndex(this->output) == maxIndex(targets[i])) {
                correct_predictions++;
            }
        }
 
        if (correct_predictions == inputs.size()) {
            std::cout << "All " << inputs.size() << " outputs in the batch are correct after " << totalEpochs << " epochs. clTraining complete." << std::endl;
            break;
        }
        else {
            std::cout << "predictions: " <<  correct_predictions << "/" << inputs.size() << std::endl;
        }
        if (totalEpochs == epochs) {
            std::cout << correct_predictions << "/" << inputs.size() << " correct. Increasing epochs by 10 and continuing clTraining." << std::endl;
            this->epochs += 10;
        }
    }
}

/**
 * @brief clTrains the mnn2d network on a single input-target pair.
 * @param input The input matrix.
 * @param target The target vector (corresponding to the pooled output).
 */
void mnn2d::clTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target) {
    int i = 0;
    while (1) {
        // 1. Forward propagation
        this->input = input;
        clForprop(this->input);

        if(maxIndex(output) == maxIndex(target)) {
            std::cout << "Correct output predicted :) at epoch " << i << "." << std::endl;
            break;
        }
        i++;

        // check for error and break if acceptable
        float loss = crossEntropy(output, target);
        std::cout << "Current CE Loss at epoch " << i << ": " <<loss << std::endl;
        if (i == EPOCH) break;

        // 2. Backward propagation
        this->target = target;
        clBackprop(this->target);
    }

    std::cout << "clTraining complete for this input-target pair." << std::endl;
}

/**
 * @brief clTrains the mnn2d network on a batch of data.
 * @param inputs A vector of input matrices.
 * @param targets A vector of target vectors.
 */
void mnn2d::clTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs and targets in batch must be the same.");
    }
    if (inputs.empty()) {
        return; // Nothing to clTrain
    }
    if (inputs[0].size() != inHeight || inputs[0][0].size() != inWidth || targets[0].size() != outWidth) {
        throw std::invalid_argument("Input or target dimensions do not match network configuration.");
    }
 
    this->batchSize = inputs.size();
    int totalEpochs = 0;
 
    while (true) {
        for (int e = 0; e < this->epochs; ++e) {
            float total_loss = 0.0f;
            for (size_t i = 0; i < inputs.size(); ++i) {
                this->input = inputs[i];
                clForprop(this->input);
                total_loss += crossEntropy(this->output, targets[i]);
            }
            totalEpochs++;
            std::cout << "Epoch " << totalEpochs << ", Average CE Loss: " << total_loss / inputs.size() << std::endl;
            clBackprop(const_cast<std::vector<std::vector<float>>&>(targets));
        }
 
        int correct_predictions = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            clForprop(inputs[i]);
            if (maxIndex(this->output) == maxIndex(targets[i])) {
                correct_predictions++;
            }
        }
 
        if (correct_predictions == inputs.size()) {
            std::cout << "All " << inputs.size() << " outputs in the batch are correct after " << totalEpochs << " epochs. clTraining complete." << std::endl;
            break;
        }
        else {
            std::cout << correct_predictions << "/" << inputs.size() << " correct. Increasing epochs by 10 and continuing clTraining." << std::endl;
            this->epochs += 10;
        }
    }
}

#endif