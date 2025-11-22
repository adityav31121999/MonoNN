#ifdef USE_CPU
#include "mnn.hpp"
#include <numeric>
#include <vector>
#include <iostream>

// Backprop for mnn

/**
 * @brief Backpropagation for the mnn class (1D data).
 * @param expected The expected output vector.
 */
void mnn::backprop(const std::vector<float>& expected) {
    int type = 3;
    this->target = expected;
    std::vector<float> output_error(outSize, 0.0f);
    for(int i = 0; i < outSize; i++) {
        output_error[i] = activate[layers-1][i] - expected[i];
    }
    std::vector<float> incoming_gradient = output_error;
    // Backpropagate the error
    for(int layer = layers - 1; layer >= 1; layer--) {
        std::vector<float> outgoing_gradient;
        layerBackward(incoming_gradient, outgoing_gradient, activate[layer-1], 
                        cweights[layer], cgradients[layer], bgradients[layer],
                        order, alpha);
        incoming_gradient = outgoing_gradient;
    }
    layerBackward(incoming_gradient, input, cweights[0], cgradients[0], bgradients[0],
                    order, alpha);
    // update weights
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
    int type = 3;
    std::vector<std::vector<float>> output_error(expected.size(), std::vector<float>(outSize, 0.0f));
    for(int i = 0; i < expected.size(); i++) {
        for(int j = 0; j < outSize; j++) {
            output_error[i][j] = outputBatch[i][j] - expected[i][j];
        }
    }
    std::vector<std::vector<float>> incoming_gradient = output_error;
    for(int layer = layers - 1; layer >= 1; layer--) {
        std::vector<std::vector<std::vector<float>>> cgrads_batch(batchSize, std::vector<std::vector<float>>(cgradients[layer].size(), std::vector<float>(cgradients[layer][0].size(), 0.0f)));
        std::vector<std::vector<std::vector<float>>> bgrads_batch(batchSize, std::vector<std::vector<float>>(bgradients[layer].size(), std::vector<float>(bgradients[layer][0].size(), 0.0f)));
        std::vector<std::vector<float>> outgoing_gradient(batchSize);
        for(int i = 0; i < batchSize; i++) {
            outgoing_gradient[i].resize(actBatch[layer-1][i].size());
            layerBackward(incoming_gradient[i], outgoing_gradient[i], actBatch[layer-1][i], 
                        cweights[layer], cgrads_batch[i], bgrads_batch[i], order, alpha);
            incoming_gradient[i] = outgoing_gradient[i];
        }
        cgradients[layer] = average(cgrads_batch);
        bgradients[layer] = average(bgrads_batch);
    }

    std::vector<std::vector<std::vector<float>>> cgrads(batchSize, std::vector<std::vector<float>>(cgradients[0].size(), std::vector<float>(cgradients[0][0].size(), 0.0f)));
    std::vector<std::vector<std::vector<float>>> bgrads(batchSize, std::vector<std::vector<float>>(bgradients[0].size(), std::vector<float>(bgradients[0][0].size(), 0.0f)));
    for(int i = 0; i < batchSize; i++) {
        layerBackward(incoming_gradient[i], inputBatch[i], cweights[0], cgrads[i], bgrads[i],
                    order, alpha);
    }
    cgradients[0] = average(cgrads);
    bgradients[0] = average(bgrads);
    // update weights
    for(int i = 0; i < layers; i++) {
        updateWeights(cweights[i], cgradients[i], learningRate, type);
        updateWeights(bweights[i], bgradients[i], learningRate, type);
    }
}

// Backprop for mnn2d

/**
 * @brief Backpropagation for the mnn2d class (2D data).
 * @param expected The expected output vector (after pooling).
 */
void mnn2d::backprop(const std::vector<float>& expected) {
    int type = 3;
    this->target = expected;
    std::vector<float> output_error(target.size(), 0.0f);
    for(int i = 0; i < outWidth; i++) {
        output_error[i] = output[i] - expected[i];
    }
    std::vector<std::vector<float>> incoming_gradient(activate[layers-1].size(), 
                                            std::vector<float>(activate[layers-1][0].size(), 0.0f));

    // Distribute output error to incoming gradient for mean pool
    for(int i = 0; i < activate[layers-1].size(); i++) {
        incoming_gradient[i] = output_error;
    }
    std::vector<std::vector<float>> outgoing_gradient;
    // Backpropagate the error
    for(int layer = layers - 1; layer >= 1; layer--) {
        layerBackward(incoming_gradient, outgoing_gradient, dotProds[layer-1], activate[layer-1],
                        cweights[layer], cgradients[layer], bgradients[layer], order, alpha);
        incoming_gradient = outgoing_gradient;
    }
    layerBackward(incoming_gradient, reshape(softmax(flatten(input)), input.size(), input[0].size()),
                    cweights[0], cgradients[0], bgradients[0], order, alpha);
    // update weights
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
    int type = 3;
    std::vector<std::vector<float>> output_error(expected.size(), std::vector<float>(expected[0].size(), 0.0f));
    for(int i = 0; i < expected.size(); i++) {
        for(int j = 0; j < expected[i].size(); j++)
            output_error[i][j] = outputBatch[i][j] - expected[i][j];
    }
    std::vector<std::vector<std::vector<float>>> incoming_gradient(batchSize);

    // Distribute output error to incoming gradient for mean pool
    for(int i = 0; i < batchSize; i++) {
        incoming_gradient[i].resize(actBatch[layers-1][i].size(), std::vector<float>(actBatch[layers-1][i][0].size()));
        for(size_t j = 0; j < actBatch[layers-1][i].size(); j++) {
            incoming_gradient[i][j] = output_error[i];
        }
    }
    // Backpropagate the error
    for(int layer = layers - 1; layer >= 1; layer--) {
        std::vector<std::vector<std::vector<float>>> cgrads(batchSize, std::vector<std::vector<float>>(cgradients[layer].size(), std::vector<float>(cgradients[layer][0].size(), 0.0f)));
        std::vector<std::vector<std::vector<float>>> bgrads(batchSize, std::vector<std::vector<float>>(bgradients[layer].size(), std::vector<float>(bgradients[layer][0].size(), 0.0f)));
        std::vector<std::vector<std::vector<float>>> outgoing_gradient(batchSize);
        for(int i = 0; i < batchSize; i++) {
            layerBackward(incoming_gradient[i], outgoing_gradient[i], dotBatch[layer-1][i], actBatch[layer-1][i],
                            cweights[layer], cgrads[i], bgrads[i], order, alpha);
            incoming_gradient[i] = outgoing_gradient[i];
        }
        cgradients[layer] = average(cgrads);
        bgradients[layer] = average(bgrads);
    }
    std::vector<std::vector<std::vector<float>>> cgrads(batchSize, std::vector<std::vector<float>>(cgradients[0].size(), std::vector<float>(cgradients[0][0].size(), 0.0f)));
    std::vector<std::vector<std::vector<float>>> bgrads(batchSize, std::vector<std::vector<float>>(bgradients[0].size(), std::vector<float>(bgradients[0][0].size(), 0.0f)));
    for(int i = 0; i < batchSize; i++) {
        layerBackward(incoming_gradient[i], reshape(softmax(flatten(inputBatch[i])), inputBatch[i].size(), inputBatch[i][0].size()),
                        cweights[0], cgrads[i], bgrads[i], order, alpha);
    }
    cgradients[0] = average(cgrads);
    bgradients[0] = average(bgrads);
    // update weights
    for(int i = 0; i < layers; i++) {
        updateWeights(cweights[i], cgradients[i], learningRate, type);
        updateWeights(bweights[i], bgradients[i], learningRate, type);
    }
}

#endif