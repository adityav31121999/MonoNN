#ifdef USE_CL
#include "mnn1d.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>

/**
 * @brief clTrains the mnn2d network on a single input-target pair.
 * @param input The input matrix.
 * @param target The target vector (corresponding to the pooled output).
 */
void mnn2d::clTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target) {
    int i = 0;
    float initialLR = this->learningRate;
    while (1) {
        // 1. Forward propagation
        this->input = softmax(input);
        clForprop(this->input);

        if(maxIndex(output) == maxIndex(target)) {
            std::cout << "Correct output predicted :) at epoch " << i << "." << std::endl;
            break;
        }
        i++;

        // check for error and break if acceptable
        currloss = crossEntropy(output, target);
        // if (i > 1) this->learningRate *= (currloss > prevloss) ? 0.95 : 1.01;
        // std::cout << "Current CE Loss at epoch " << i << " : " << currloss << std::endl;
                  // << "\nOpting for new learning rate: " << this->learningRate << std::endl;

        // if (i == EPOCH) break;
        // 2. Backward propagation
        this->target = target;
        zeroGradients();
        clBackprop(this->target);
    }

    this->learningRate = initialLR; // reset learning rate after clTraining
    // std::cout << "Training complete for this input-target pair." << std::endl;
}


/**
 * @brief clTrains the mnn2d network on a batch of data (input-target pairs).
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
    if (inputs[0].size() != inHeight || inputs[0][0].size() != inWidth || targets[0].size() != outSize) {
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
        targetBatch.resize(batchSize);
        for(int i=0; i<batchSize; ++i) outputBatch[i].resize(outSize);
        for(int i=0; i<batchSize; ++i) targetBatch[i].resize(outSize);
    }

    int totalEpochs = 0;
    if (this->epochs < 1) this->epochs = 100;
    std::vector<int> correct(input.size(), -1);

    while (true) {
        float total_loss = 0.0f;
        clForprop(inputBatch); // Batch clForprop
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
            std::cout << "All " << inputs.size() << " outputs in the batch are correct after " << totalEpochs 
                      << " epochs. Training complete with error " << currloss << "." << std::endl;
            break;
        }
        else {
            std::cout << "Correct Predictions: ";
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::cout << correct[i] << " ";
            }
            // loss calculation
            for (size_t i = 0; i < inputs.size(); ++i) {
                total_loss += crossEntropy(outputBatch[i], targets[i]);
            }
            currloss = static_cast<float>(total_loss / BATCH_SIZE);
            std::cout << "-> Epoch: " << totalEpochs << "\tPredictions: " << correct_predictions << "/" << inputs.size() 
                      << "\tAvg. CE Loss: " << currloss << std::endl;
        }

        if (totalEpochs >= this->epochs) {
            std::cout << correct_predictions << "/" << inputs.size() << " correct. Increasing epochs by 10 and continuing clTraining." << std::endl;
            this->epochs += 10;
        }
        zeroGradients();
        clBackprop(targets);
    }
}

#endif