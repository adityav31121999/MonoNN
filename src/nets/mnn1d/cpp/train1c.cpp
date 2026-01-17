#ifdef USE_CPU
#include "mnn1d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>

/**
 * @brief trains the mnn network on a single input-target pair for 1 cycle.
 * @param input The input vector.
 * @param target The target output vector.
 * @param useThread 1 to use thread based faster execution else 0.
 */
void mnn1d::train1c(const std::vector<float>& input, const std::vector<float>& target, bool useThread) {
    // single cycle training
    if (useThread == 0) {
        // 1. Forward propagation
        forprop(input);

        if(maxIndex(output) == maxIndex(target)) {
            float loss = crossEntropy(output, target);
            if (loss < 0) loss = 0;
            // std::cout << "Correct output predicted with loss " << loss << "." << std::endl;
        }
        else {
            // check for error and break if acceptable
            currloss = crossEntropy(output, target);
            // std::cout << "Current CE Loss: " << currloss << std::endl;

            // 2. Backward propagation
            backprop(target);
            prevloss = currloss;
        }
    }
    else {
        layerForwardThread(input, dotProds[0], cweights[0], bweights[0], order);
        activate[0] = sigmoid(dotProds[0]);
        // from 2nd to last
        for(int i = 1; i < layers; i++) {
            layerForwardThread(activate[i-1], dotProds[i], cweights[i], bweights[i], order);
            activate[i] = sigmoid(dotProds[i]);
        }
        output = softmax(activate[layers - 1]);

        if(maxIndex(output) == maxIndex(target)) {
            float loss = crossEntropy(output, target);
            if (loss < 0) loss = 0;
            // std::cout << "Correct output predicted with loss " << loss << "." << std::endl;
        }
        else {
            // check for error and break if acceptable
            currloss = crossEntropy(output, target);
            // std::cout << "Current CE Loss: " << currloss << std::endl;

            // 2. Backward propagation
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
        }
    }
}

#endif