#ifdef USE_CPU
#include "mnn.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>

// zero out gradients before backprop
void mnn::zeroGradients() {
    for(auto& layer : cgradients) {
        for(auto& row : layer) std::fill(row.begin(), row.end(), 0.0f);
    }
    for(auto& layer : bgradients) {
        for(auto& row : layer) std::fill(row.begin(), row.end(), 0.0f);
    }
}


/**
 * @brief trains the mnn network on a single input-target pair.
 * @param input The input vector.
 * @param target The target output vector.
 */
void mnn::train(const std::vector<float>& input, const std::vector<float>& target) {
    int i = 0;
    while (1) {
        zeroGradients();
        // 1. Forward propagation
        this->input = input;
        forprop(this->input);

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
        backprop(this->target);
    }

    std::cout << "Training complete for this input-target pair." << std::endl;
}

/**
 * @brief trains the mnn network on a batch of data (input-target pairs).
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
        targetBatch.resize(batchSize);
        for(int i=0; i<batchSize; ++i) outputBatch[i].resize(outSize);
        for(int i=0; i<batchSize; ++i) targetBatch[i].resize(outSize);
    }

    int totalEpochs = 0;
    if (this->epochs < 1) this->epochs = 100;
    std::vector<int> correct(input.size(), -1);
 
    // inputBatch = inputs;
    while (true) {
        zeroGradients();
        float total_loss = 0.0f;
        forprop(inputBatch);
        for (size_t i = 0; i < inputs.size(); ++i) {
            total_loss += crossEntropy(outputBatch[i], targets[i]);
        }
        totalEpochs++;
 
        int correct_predictions = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Use outputBatch for checking accuracy to avoid re-computation
            if (maxIndex(outputBatch[i]) == maxIndex(targets[i])) {
                correct_predictions++;
                correct[i] = 1;
            }
            else
                correct[i] = 0;
        }

        if (correct_predictions == inputs.size()) {
            std::cout << "All " << inputs.size() << " outputs in the batch are correct after " << totalEpochs << " epochs. training complete." << std::endl;
            break;
        }
        else {
            std::cout << "Correct Predictions: ";
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::cout << correct[i] << " ";
            }
            std::cout << "\nEpoch: " << totalEpochs << " \t|\t predictions: " << correct_predictions << "/" << inputs.size() 
                      << " \t|\t Average CE Loss: " << static_cast<float>(total_loss / batchSize) << std::endl;
        }

        if (totalEpochs >= this->epochs) {
            std::cout << correct_predictions << "/" << inputs.size() << " correct. Increasing epochs by 10 and continuing training." << std::endl;
            this->epochs += 10;
        }
        // targetBatch = targets;
        backprop(targets);
    }
}


/**
 * @brief trains the mnn2d network on a single input-target pair.
 * @param input The input matrix.
 * @param target The target vector (corresponding to the pooled output).
 */
void mnn2d::train(const std::vector<std::vector<float>>& input, const std::vector<float>& target) {
    int i = 0;
    while (1) {
        // 1. Forward propagation
        this->input = input;
        forprop(this->input);

        if(maxIndex(output) == maxIndex(target)) {
            std::cout << "Correct output predicted :) at epoch " << i << "." << std::endl;
            break;
        }
        i++;

        // check for error and break if acceptable
        float loss = crossEntropy(output, target);
        std::cout << "Current CE Loss at epoch " << i << ": " << loss << std::endl;

        if (i == EPOCH) break;

        // 2. Backward propagation
        this->target = target;
        backprop(this->target);
    }

    std::cout << "Training complete for this input-target pair." << std::endl;
}

/**
 * @brief trains the mnn2d network on a batch of data (input-target pairs).
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
    std::vector<int> correct(input.size(), -1);

    inputBatch = inputs;
    while (true) {
        float total_loss = 0.0f;
        forprop(inputBatch); // Batch forprop
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

        int correct_predictions = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Use outputBatch for checking accuracy to avoid re-computation
            if (maxIndex(outputBatch[i]) == maxIndex(targets[i])) {
                correct_predictions++;
                correct[i] = 1;
            }
            else
                correct[i] = 0;
        }

        if (correct_predictions == inputs.size()) {
            std::cout << "All " << inputs.size() << " outputs in the batch are correct after " << totalEpochs << " epochs. training complete." << std::endl;
            break;
        }
        else {
            std::cout << "Correct Predictions: ";
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::cout << correct[i] << " ";
            }
            std::cout << "\nEpoch: " << totalEpochs << " \t|\t predictions: " << correct_predictions << "/" << inputs.size() << " \t|\t Average CE Loss: " << total_loss / inputs.size() << std::endl;
        }

        if (totalEpochs >= this->epochs) {
            std::cout << correct_predictions << "/" << inputs.size() << " correct. Increasing epochs by 10 and continuing training." << std::endl;
            this->epochs += 10;
        }
        backprop(targets);
    }
}

#endif