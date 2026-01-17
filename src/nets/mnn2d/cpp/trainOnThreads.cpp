#ifdef USE_CPU
#include "mnn1d.hpp"
#include "mnn2d.hpp"

void mnn2d::threadTrain(const std::vector<std::vector<float>> &input, const std::vector<float> &target)
{
    int i = 0;
    float initialLR = this->learningRate;
    while (1) {
        // 1. Forward propagation
        this->input = softmax(input);
        // first layer
        layerForward(input, dotProds[0], cweights[0], bweights[0], order);
        activate[0] = relu(dotProds[0]);
        // from 2nd to last
        for(int i = 1; i < layers; i++) {
            layerForward(activate[i-1], dotProds[i], cweights[i], bweights[i], order);
            activate[i] = relu(dotProds[i]);
        }
        // apply mean pooling to the final activation layer to get output
        output = softmax(meanPool(activate[layers - 1]));

        if(maxIndex(output) == maxIndex(target)) {
            std::cout << "Correct output predicted :) at epoch " << i << "." << std::endl;
            break;
        }
        i++;

        // check for error and break if acceptable
        currloss = crossEntropy(output, target);
        // if (i > 1) this->learningRate *= (currloss > prevloss) ? 0.95 : 1.01;
        std::cout << "Current CE Loss at epoch " << i << " : " << currloss << std::endl;
                  // << "\nOpting for new learning rate: " << this->learningRate << std::endl;

        // if (i == EPOCH) break;
        // 2. Backward propagation
        this->target = target;
        zeroGradients();
        std::vector<float> meanpooled(target.size(), 0.0f);
        meanpooled = meanPool(activate[layers-1]);
        std::vector<float> output_error(target.size(), 0.0f);
        for(int i = 0; i < outSize; i++) {
            output_error[i] = meanpooled[i] - target[i];
            output_error[i] /= inHeight;
        }
        // output was mean pooled from activate[layers-1]
        std::vector<std::vector<float>> incoming_gradient(activate[layers-1].size(), 
                                                std::vector<float>(activate[layers-1][0].size(), 0.0f));
        for(int i = 0; i < activate[layers-1].size(); i++) {
            for(int j = 0; j < outSize; j++) {
                incoming_gradient[i][j] = output_error[j];
            }
        }
        std::vector<std::vector<float>> outgoing_gradient;

        // Backpropagate the error
        for(int layer = layers - 1; layer >= 1; layer--) {
            layerBackwardThread(incoming_gradient, outgoing_gradient, dotProds[layer-1], activate[layer-1],
                            cweights[layer], cgradients[layer], bgradients[layer], order, ALPHA);
            incoming_gradient.clear();
            incoming_gradient = outgoing_gradient;
        }
        layerBackwardThread(incoming_gradient, input, cweights[0], cgradients[0], bgradients[0], order, ALPHA);

        // update weights
        for(int i = 0; i < layers; i++) {
            updateWeights(cweights[i], cgradients[i], learningRate, weightUpdateType);
            updateWeights(bweights[i], bgradients[i], learningRate, weightUpdateType);
        }
    }

    this->learningRate = initialLR; // reset learning rate after training
    std::cout << "Training complete for this input-target pair." << std::endl;
}


void mnn2d::threadTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>> &targets)
{
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
    this->learningRate = initialLR; // reset learning rate after training
}

#endif // USE_CPU