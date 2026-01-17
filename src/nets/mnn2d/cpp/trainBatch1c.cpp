#ifdef USE_CPU
#include "mnn1d.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>

/**
 * @brief trains the mnn2d network on a batch of data for single cycle.
 * @param inputs A vector of input matrices.
 * @param targets A vector of target vectors.
 * @param useThread 1 to use thread based faster execution else 0.
 */
void mnn2d::trainBatch1c(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets, bool useThread) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs and targets in batch must be the same.");
    }
    if (inputs.empty()) {
        return; // Nothing to train
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
        for(int i=0; i<batchSize; ++i) outputBatch[i].resize(outSize);
    }

    if (this->epochs < 1) this->epochs = 100;
    std::vector<int> correct(input.size(), -1);
    float total_loss = 0.0f;

    if (useThread == 0) {
        forprop(inputs); // Batch forprop

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
            std::cout << "All " << inputs.size() << " outputs in the batch are correct. training complete." << std::endl;
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
            currloss = static_cast<float>(total_loss / batchSize);
            std::cout << " | Predictions: " << correct_predictions << "/" << inputs.size() 
                        << "\tAvg. CE Loss: " << currloss << std::endl;
            backprop(targets);
        }
    }
    else {
        // first layer
        layerForwardBatchThread(inputBatch, dotBatch[0], cweights[0], bweights[0], order);
        for(int i = 0; i < batchSize; i++) {
            actBatch[0][i] = relu(dotBatch[0][i]);
        }

        // from 2nd to last
        for(int j = 1; j < layers; j++) {
            layerForwardBatchThread(actBatch[j-1], dotBatch[j], cweights[j], bweights[j], order);
            for(int i = 0; i < batchSize; i++) {
                actBatch[j][i] = relu(dotBatch[j][i]);
            }
        }

        // assign output batch
        for(int i = 0; i < batchSize; i++) {
            outputBatch[i] = softmax(meanPool(actBatch[layers-1][i]));
        }

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
            std::cout << "All " << inputs.size() << " outputs in the batch are correc. training complete." << std::endl;
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
            currloss = static_cast<float>(total_loss / batchSize);
            std::cout << " | Predictions: " << correct_predictions << "/" << inputs.size() 
                        << "\tAvg. CE Loss: " << currloss << std::endl;

            zeroGradients();
            std::vector<std::vector<float>> meanpooled(targets.size(), std::vector<float>(targets[0].size(), 0.0f));
            for(int i; i < targets.size(); i++) {
                meanpooled[i] = meanPool(actBatch[layers-1][i]);
            }
            std::vector<std::vector<float>> output_error(targets.size(), std::vector<float>(targets[0].size(), 0.0f));
            for(int i = 0; i < targets.size(); i++) {
                for(int j = 0; j < targets[i].size(); j++) {
                    output_error[i][j] = meanpooled[i][j] - targets[i][j];
                }
            }

            // Distribute output error to incoming gradient for mean pool
            std::vector<std::vector<std::vector<float>>> incoming_gradient(batchSize);
            for(int i = 0; i < batchSize; i++) {
                incoming_gradient[i].resize(actBatch[layers-1][i].size(), std::vector<float>(actBatch[layers-1][i][0].size()));
                for(size_t j = 0; j < actBatch[layers-1][i].size(); j++) {
                    for(size_t k = 0; k < outSize; k++) {
                        incoming_gradient[i][j][k] = output_error[i][k];
                    }
                }
            }

            // Backpropagate the error
            for(int layer = layers - 1; layer >= 1; layer--) {
                std::vector<std::vector<std::vector<float>>> outgoing_gradient;
                layerBackwardBatchThread(incoming_gradient, outgoing_gradient, dotBatch[layer-1], actBatch[layer-1],
                                cweights[layer], cgradients[layer], bgradients[layer], order, ALPHA);
                incoming_gradient.clear();
                incoming_gradient = outgoing_gradient;
            }
            layerBackwardBatchThread(incoming_gradient, inputBatch, cweights[0], cgradients[0], bgradients[0], order, ALPHA);

            // update weights
            for(int i = 0; i < layers; i++) {
                updateWeights(cweights[i], cgradients[i], learningRate, weightUpdateType);
                updateWeights(bweights[i], bgradients[i], learningRate, weightUpdateType);
            }
        }
    }
}

#endif // USE_CPU