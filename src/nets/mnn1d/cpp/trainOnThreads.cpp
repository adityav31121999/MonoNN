#ifdef USE_CPU
#include "mnn1d.hpp"

void mnn1d::threadTrain(const std::vector<float> &input, const std::vector<float> &target)
{
    int i = 0;
    float initialLR = this->learningRate;
    while (1) {
        // 1. Forward propagation
        this->input = softmax(input);
        layerForwardThread(input, dotProds[0], cweights[0], bweights[0], order);
        activate[0] = sigmoid(dotProds[0]);
        // from 2nd to last
        for(int i = 1; i < layers; i++) {
            layerForwardThread(activate[i-1], dotProds[i], cweights[i], bweights[i], order);
            activate[i] = sigmoid(dotProds[i]);
        }
        output = softmax(activate[layers - 1]);

        if(maxIndex(output) == maxIndex(target)) {
            std::cout << "Correct output predicted :) at epoch " << i << " with loss " << crossEntropy(output, target) << "." << std::endl;
            break;
        }
        i++;

        // check for error and break if acceptable
        currloss = crossEntropy(output, target);
        // if (i > 1) this->learningRate *= (currloss > prevloss) ? 0.95 : 1.01;
        std::cout << "Current CE Loss at epoch " << i << " : " << currloss << std::endl;
                  // << "\nOpting for new learning rate: " << this->learningRate << std::endl;

        // 2. Backward propagation
        this->target = target;
        zeroGradients();
        std::vector<float> output_error(outSize, 0.0f);
        for(int i = 0; i < outSize; i++) {
            output_error[i] = activate[layers-1][i] - target[i];
        }
        std::vector<float> incoming_gradient = output_error;
        // Backpropagate the error
        for(int layer = layers - 1; layer >= 1; layer--) {
            std::vector<float> outgoing_gradient;
            layerBackwardThread(incoming_gradient, outgoing_gradient, activate[layer-1], cweights[layer], cgradients[layer],
                            bgradients[layer], order, ALPHA);
            incoming_gradient.clear();
            incoming_gradient = outgoing_gradient;
        }
        layerBackwardThread(incoming_gradient, input, cweights[0], cgradients[0], bgradients[0], order, ALPHA);

        // update weights
        for(int i = 0; i < layers; i++) {
            updateWeights(cweights[i], cgradients[i], learningRate, weightUpdateType);
            updateWeights(bweights[i], bgradients[i], learningRate, weightUpdateType);
        }
        prevloss = currloss;
        // if (i == EPOCH) break;
    }

    this->learningRate = initialLR; // reset learning rate after training
    std::cout << "Training complete for this input-target pair." << std::endl;
}

void mnn1d::threadTrainBatch(const std::vector<std::vector<float>> &inputs, const std::vector<std::vector<float>> &targets)
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
        for(int i = 0; i < batchSize; i++) {
            outputBatch[i] = softmax(actBatch[layers-1][i]);
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
                output_error[i][j] = actBatch[layers-1][i][j] - targets[i][j];
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

#endif // USE_CPU