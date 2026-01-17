#ifdef USE_CPU
#include "mnn1d.hpp"
#include "mnn2d.hpp"
#include <numeric>
#include <vector>
#include <iostream>

// Backprop for mnn

/**
 * @brief Backpropagation for the mnn class (1D data).
 * @param expected The expected output vector.
 */
void mnn1d::backprop(const std::vector<float>& expected) {
    zeroGradients();
    std::vector<float> output_error(outSize, 0.0f);
    for(int i = 0; i < outSize; i++) {
        output_error[i] = activate[layers-1][i] - expected[i];
    }
    std::vector<float> incoming_gradient = output_error;
    // Backpropagate the error
    for(int layer = layers - 1; layer >= 1; layer--) {
        std::vector<float> outgoing_gradient;
        layerBackward(incoming_gradient, outgoing_gradient, activate[layer-1], cweights[layer], cgradients[layer],
                        bgradients[layer], order, ALPHA);
        incoming_gradient.clear();
        incoming_gradient = outgoing_gradient;
    }
    layerBackward(incoming_gradient, input, cweights[0], cgradients[0], bgradients[0], order, ALPHA);

    // update weights
    for(int i = 0; i < layers; i++) {
        updateWeights(cweights[i], cgradients[i], learningRate, weightUpdateType);
        updateWeights(bweights[i], bgradients[i], learningRate, weightUpdateType);
    }
}


/**
 * @brief Backpropagation for the mnn class (1D data) for batch backpropagation
 *  by discrete gradient calculation and averaging final gradients.
 * @param expected The expected output vector.
 */
void mnn1d::backprop(const std::vector<std::vector<float>>& expected)
{
    zeroGradients();
    std::vector<std::vector<float>> output_error(expected.size(), std::vector<float>(outSize, 0.0f));
    for(int i = 0; i < expected.size(); i++) {
        for(int j = 0; j < outSize; j++) {
            output_error[i][j] = actBatch[layers-1][i][j] - expected[i][j];
        }
    }
    
    std::vector<std::vector<float>> incoming_gradient = output_error;
    
    for(int layer = layers - 1; layer >= 1; layer--) {
        std::vector<std::vector<float>> outgoing_gradient;
        layerBackwardBatch(incoming_gradient, outgoing_gradient, actBatch[layer-1],
                          cweights[layer], cgradients[layer], bgradients[layer], order, ALPHA);
        incoming_gradient = std::move(outgoing_gradient);
    }
    
    layerBackwardBatch(incoming_gradient, inputBatch, cweights[0], cgradients[0], bgradients[0],
                              order, ALPHA);
    
    // update weights
    for(int i = 0; i < layers; i++) {
        updateWeights(cweights[i], cgradients[i], learningRate, weightUpdateType);
        updateWeights(bweights[i], bgradients[i], learningRate, weightUpdateType);
    }
}

#endif