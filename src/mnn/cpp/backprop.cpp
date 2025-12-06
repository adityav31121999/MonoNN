#ifdef USE_CPU
#include "mnn.hpp"
#include "mnn2d.hpp"
#include <numeric>
#include <vector>
#include <iostream>

// Backprop for mnn

/**
 * @brief Backpropagation for the mnn class (1D data).
 * @param expected The expected output vector.
 */
void mnn::backprop(const std::vector<float>& expected) {
    this->target = expected;
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
    int type = 3;
    for(int i = 0; i < layers; i++) {
        updateWeights(cweights[i], cgradients[i], learningRate, type);
        updateWeights(bweights[i], bgradients[i], learningRate, type);
    }
}


/**
 * @brief Backpropagation for the mnn class (1D data) for batch backpropagation
 *  by discrete gradient calculation and averaging final gradients.
 * @param expected The expected output vector.
 */
void mnn::backprop(const std::vector<std::vector<float>>& expected)
{    
    std::vector<std::vector<float>> output_error(expected.size(), std::vector<float>(outSize, 0.0f));
    for(int i = 0; i < expected.size(); i++) {
        for(int j = 0; j < outSize; j++) {
            output_error[i][j] = outputBatch[i][j] - expected[i][j];
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
    int type = 3;
    for(int i = 0; i < layers; i++) {
        updateWeights(cweights[i], cgradients[i], learningRate, type);
        updateWeights(bweights[i], bgradients[i], learningRate, type);
    }
    zeroGradients();
}

// Backprop for mnn2d

/**
 * @brief Backpropagation for the mnn2d class (2D data).
 * @param expected The expected output vector (after pooling).
 */
void mnn2d::backprop(const std::vector<float>& expected) {
    this->target = expected;
    std::vector<float> output_error(target.size(), 0.0f);
    for(int i = 0; i < outWidth; i++) {
        output_error[i] = output[i] - expected[i];
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
        layerBackward(incoming_gradient, outgoing_gradient, dotProds[layer-1], activate[layer-1],
                        cweights[layer], cgradients[layer], bgradients[layer], order, ALPHA);
        incoming_gradient.clear();
        incoming_gradient = outgoing_gradient;
    }
    layerBackward(incoming_gradient, input, cweights[0], cgradients[0], bgradients[0], order, ALPHA);

    // update weights
    int type = 3;
    for(int i = 0; i < layers; i++) {
        updateWeights(cweights[i], cgradients[i], learningRate, type);
        updateWeights(bweights[i], bgradients[i], learningRate, type);
    }
}

/**
 * @brief Backpropagation for the mnn2d class (2D data) for batch backpropagation
 *  by discrete gradient calculation and averaging final gradients.
 * @param expected The expected output vector (after pooling).
 */
void mnn2d::backprop(const std::vector<std::vector<float>>& expected) {
    std::vector<std::vector<float>> output_error(expected.size(), std::vector<float>(expected[0].size(), 0.0f));
    for(int i = 0; i < expected.size(); i++) {
        for(int j = 0; j < expected[i].size(); j++) {
            output_error[i][j] = outputBatch[i][j] - expected[i][j];
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
        layerBackwardBatch(incoming_gradient, outgoing_gradient, dotBatch[layer-1], actBatch[layer-1],
                        cweights[layer], cgradients[layer], bgradients[layer], order, ALPHA);
        incoming_gradient.clear();
        incoming_gradient = outgoing_gradient;
    }
    layerBackwardBatch(incoming_gradient, inputBatch, cweights[0], cgradients[0], bgradients[0], order, ALPHA);

    // update weights
    int type = 3;
    for(int i = 0; i < layers; i++) {
        updateWeights(cweights[i], cgradients[i], learningRate, type);
        updateWeights(bweights[i], bgradients[i], learningRate, type);
    }
}

#endif