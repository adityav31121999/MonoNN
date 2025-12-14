#ifdef USE_CPU
#include "mnn.hpp"
#include "mnn2d.hpp"

void mnn::threadTrainBatch(const std::vector<std::vector<float>> &inputs, const std::vector<std::vector<float>> &targets)
{
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
    inputBatch = inputs;
    float initialLR = this->learningRate;
    for (size_t i = 0; i < inputs.size(); ++i) {
        this->inputBatch[i] = softmax(inputs[i]);
    }

    while (true) {
        float total_loss = 0.0f;
        // first layer
        layerForwardBatchThread(inputBatch, dotBatch[0], cweights[0], bweights[0], order);
        for(int i = 0; i < batchSize; i++) {
            actBatch[0][i] = sigmoid(dotBatch[0][i]);
        }
        // from 2nd to last
        for(int j = 1; j < layers; j++) {
            layerForwardBatchThread(actBatch[j-1], dotBatch[j], cweights[j], bweights[j], order);
            for(int i = 0; i < batchSize; i++) {
                actBatch[j][i] = sigmoid(dotBatch[j][i]);
            }
        }
        outputBatch = actBatch[layers-1];

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
            std::cout << "Correct Predictions: ";
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::cout << correct[i] << " ";
            }
            // loss calculation
            for (size_t i = 0; i < inputs.size(); ++i) {
                total_loss += crossEntropy(outputBatch[i], targets[i]);
            }
            currloss = static_cast<float>(total_loss / BATCH_SIZE);
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
            std::cout << correct_predictions << "/" << inputs.size() << " correct. Increasing epochs by 50 and continuing training." << std::endl;
            this->epochs += 50;
        }

        targetBatch = targets;
        zeroGradients();
        std::vector<std::vector<float>> output_error(targets.size(), std::vector<float>(outSize, 0.0f));
        for(int i = 0; i < targets.size(); i++) {
            for(int j = 0; j < outSize; j++) {
                output_error[i][j] = outputBatch[i][j] - targets[i][j];
            }
        }
        
        std::vector<std::vector<float>> incoming_gradient = output_error;
        for(int layer = layers - 1; layer >= 1; layer--) {
            std::vector<std::vector<float>> outgoing_gradient;
            layerBackwardBatchThread(incoming_gradient, outgoing_gradient, actBatch[layer-1],
                            cweights[layer], cgradients[layer], bgradients[layer], order, ALPHA);
            incoming_gradient = std::move(outgoing_gradient);
        }
        
        layerBackwardBatchThread(incoming_gradient, inputBatch, cweights[0], cgradients[0], bgradients[0],
                                order, ALPHA);
    
        // update weights
        for(int i = 0; i < layers; i++) {
            updateWeights(cweights[i], cgradients[i], learningRate, weightUpdateType);
            updateWeights(bweights[i], bgradients[i], learningRate, weightUpdateType);
        }
    }
    this->learningRate = initialLR; // reset learning rate after training
}


void mnn2d::threadTrainBatch(const std::vector<std::vector<std::vector<float>>> &inputs, const std::vector<std::vector<float>> &targets)
{
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
    float initialLR = this->learningRate;
    inputBatch = inputs;

    while (true) {
        float total_loss = 0.0f;
        // first layer
        layerForwardBatchThread(inputBatch, dotBatch[0], cweights[0], bweights[0], order);
        for(int i = 0; i < batchSize; i++) {
            actBatch[0][i] = reshape(softmax(flatten(dotBatch[0][i])), dotBatch[0][i].size(), dotBatch[0][i][0].size());
        }

        // from 2nd to last
        for(int j = 1; j < layers; j++) {
            layerForwardBatchThread(actBatch[j-1], dotBatch[j], cweights[j], bweights[j], order);
            for(int i = 0; i < batchSize; i++) {
                actBatch[j][i] = reshape(softmax(flatten(dotBatch[j][i])), dotBatch[j][i].size(), dotBatch[j][i][0].size());
            }
        }

        // assign output batch
        for(int i = 0; i < batchSize; i++) {
            outputBatch[i] = meanPool(actBatch[layers-1][i]);
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
            // loss calculation
            for (size_t i = 0; i < inputs.size(); ++i) {
                total_loss += crossEntropy(outputBatch[i], targets[i]);
            }
            currloss = static_cast<float>(total_loss / batchSize);
            std::cout << " | Epoch: " << totalEpochs << "\tPredictions: " << correct_predictions << "/" << inputs.size() 
                      << "\tAvg. CE Loss: " << currloss << std::endl;
        }

        if (totalEpochs >= this->epochs) {
            std::cout << correct_predictions << "/" << inputs.size() << " correct. Increasing epochs by 10 and continuing training." << std::endl;
            this->epochs += 10;
        }
        zeroGradients();
        std::vector<std::vector<float>> output_error(targets.size(), std::vector<float>(targets[0].size(), 0.0f));
        for(int i = 0; i < targets.size(); i++) {
            for(int j = 0; j < targets[i].size(); j++) {
                output_error[i][j] = outputBatch[i][j] - targets[i][j];
            }
        }
        std::vector<std::vector<std::vector<float>>> incoming_gradient(batchSize);

        // Distribute output error to incoming gradient for mean pool
        for(int i = 0; i < batchSize; i++) {
            incoming_gradient[i].resize(actBatch[layers-1][i].size(), std::vector<float>(actBatch[layers-1][i][0].size()));
            for(size_t j = 0; j < actBatch[layers-1][i].size(); j++) {
                for(size_t k = 0; k < outWidth; k++) {
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
    this->learningRate = initialLR; // reset learning rate after training
}

#endif // USE_CPU