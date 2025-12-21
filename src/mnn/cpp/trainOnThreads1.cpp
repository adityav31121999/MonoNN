#ifdef USE_CPU
#include "mnn1d.hpp"
#include "mnn2d.hpp"

void mnn::threadTrain(const std::vector<float> &input, const std::vector<float> &target)
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
        output = activate[layers - 1];

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


void mnn2d::threadTrain(const std::vector<std::vector<float>> &input, const std::vector<float> &target)
{
    int i = 0;
    float initialLR = this->learningRate;
    while (1) {
        // 1. Forward propagation
        this->input = softmax(input);
        // first layer
        layerForward(input, dotProds[0], cweights[0], bweights[0], order);
        activate[0] = reshape(softmax(flatten(dotProds[0])), dotProds[0].size(), dotProds[0][0].size());
        // from 2nd to last
        for(int i = 1; i < layers; i++) {
            layerForward(activate[i-1], dotProds[i], cweights[i], bweights[i], order);
            activate[i] = reshape(softmax(flatten(dotProds[i])), dotProds[i].size(), dotProds[i][0].size());
        }
        // apply mean pooling to the final activation layer to get output
        output = meanPool(activate[layers - 1]);

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
        std::vector<float> output_error(target.size(), 0.0f);
        for(int i = 0; i < outWidth; i++) {
            output_error[i] = output[i] - target[i];
        }
        // output was mean pooled from activate[layers-1]
        std::vector<std::vector<float>> incoming_gradient(activate[layers-1].size(), 
                                                std::vector<float>(activate[layers-1][0].size(), 0.0f));
        for(int i = 0; i < activate[layers-1].size(); i++) {
            for(int j = 0; j < outWidth; j++) {
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

#endif // USE_CPU