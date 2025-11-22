#ifdef USE_CL
#include "mnn.hpp"
#include "mnn2d.hpp"
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

        if(maxIndex(output) == maxIndex(target)) {
            std::cout << "Correct output predicted :) at epoch " << i << " with loss " << crossEntropy(output, target) << "." << std::endl;
            break;
        }
        i++;

        // check for error and break if acceptable
        float loss = crossEntropy(output, target);
        std::cout << "Current CE Loss at epoch " << i << " : " <<loss << std::endl;
/*
        // Log diagnostic statistics every 10 epochs
        if (i % 20 == 0) {
            std::cout << "=== Diagnostic Statistics at Epoch " << i << " ===" << std::endl;
            computeStats(cweights, bweights, cgradients, bgradients, activate);
        }
*/
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

    // Resize batch vectors
    if (dotBatch.size() != layers) {
        dotBatch.resize(layers);
        actBatch.resize(layers);
    }
    for (int i = 0; i < layers; ++i) {
        if (dotBatch[i].size() != batchSize) {
            dotBatch[i].resize(batchSize);
            actBatch[i].resize(batchSize);
            for (int j = 0; j < batchSize; ++j) {
                dotBatch[i][j].resize(width[i]);
                actBatch[i][j].resize(width[i]);
            }
        }
    }
    if (outputBatch.size() != batchSize) {
        outputBatch.resize(batchSize);
        for(int i=0; i<batchSize; ++i) outputBatch[i].resize(outSize);
    }

    int totalEpochs = 0;
    if (this->epochs < 1) this->epochs = 100;
 
    while (true) {
        float total_loss = 0.0f;
        inputBatch = inputs;
        clForprop(inputs);
        for (size_t i = 0; i < inputs.size(); ++i) {
            total_loss += crossEntropy(outputBatch[i], targets[i]);
        }
        totalEpochs++;
/*
        // Log diagnostic statistics every 10 epochs
        if (totalEpochs % 20 == 0) {
            std::cout << "=== Diagnostic Statistics at Epoch " << totalEpochs << " ===" << std::endl;
            computeStats(cweights, bweights, cgradients, bgradients, activate);
        }
*/
        clBackprop(const_cast<std::vector<std::vector<float>>&>(targets));
 
        int correct_predictions = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Use outputBatch for checking accuracy to avoid re-computation
            if (maxIndex(outputBatch[i]) == maxIndex(targets[i])) {
                correct_predictions++;
            }
        }

        if (correct_predictions == inputs.size()) {
            std::cout << "All " << inputs.size() << " outputs in the batch are correct after " << totalEpochs << " epochs. clTraining complete." << std::endl;
            break;
        }
        else {
            std::cout << "Epoch: " << totalEpochs << " \t|\t predictions: " << correct_predictions << "/" << inputs.size() << " \t|\t Average CE Loss: " << total_loss / inputs.size() << std::endl;
        }

        if (totalEpochs >= this->epochs) {
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
        std::cout << "Current CE Loss at epoch " << i << ": " << loss << std::endl;
/*
        // Log diagnostic statistics every 10 epochs
        if (i % 20 == 0) {
            std::cout << "=== Diagnostic Statistics at Epoch " << i << " ===" << std::endl;
            computeStats(cweights, bweights, cgradients, bgradients, activate);
        }
*/
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

    // Resize batch vectors
    if (dotBatch.size() != layers) {
        dotBatch.resize(layers);
        actBatch.resize(layers);
    }
    for (int i = 0; i < layers; ++i) {
        if (dotBatch[i].size() != batchSize) {
            dotBatch[i].resize(batchSize);
            actBatch[i].resize(batchSize);
            for (int j = 0; j < batchSize; ++j) {
                int cols = width[i];
                dotBatch[i][j].resize(inHeight, std::vector<float>(cols));
                actBatch[i][j].resize(inHeight, std::vector<float>(cols));
            }
        }
    }
    if (outputBatch.size() != batchSize) {
        outputBatch.resize(batchSize);
        for(int i=0; i<batchSize; ++i) outputBatch[i].resize(outWidth);
    }

    int totalEpochs = 0;
    if (this->epochs < 1) this->epochs = 100;

    while (true) {
        float total_loss = 0.0f;
        inputBatch = inputs;
        clForprop(inputs); // Batch clForprop
        for (size_t i = 0; i < inputs.size(); ++i) {
            total_loss += crossEntropy(outputBatch[i], targets[i]);
        }
        totalEpochs++;
/*
        // Log diagnostic statistics every 10 epochs
        if (totalEpochs % 20 == 0) {
            std::cout << "=== Diagnostic Statistics at Epoch " << totalEpochs << " ===" << std::endl;
            computeStats(cweights, bweights, cgradients, bgradients, activate);
        }
*/
        clBackprop(const_cast<std::vector<std::vector<float>>&>(targets));

        int correct_predictions = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Use outputBatch for checking accuracy to avoid re-computation
            if (maxIndex(outputBatch[i]) == maxIndex(targets[i])) {
                correct_predictions++;
            }
        }

        if (correct_predictions == inputs.size()) {
            std::cout << "All " << inputs.size() << " outputs in the batch are correct after " << totalEpochs << " epochs. clTraining complete." << std::endl;
            break;
        }
        else {
            std::cout << "Epoch: " << totalEpochs << " \t|\t predictions: " << correct_predictions << "/" << inputs.size() << " \t|\t Average CE Loss: " << total_loss / inputs.size() << std::endl;
        }

        if (totalEpochs >= this->epochs) {
            std::cout << correct_predictions << "/" << inputs.size() << " correct. Increasing epochs by 10 and continuing clTraining." << std::endl;
            this->epochs += 10;
        }
    }
}

#endif