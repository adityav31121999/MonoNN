#ifdef USE_CPU
#include "mnn.hpp"
#include <numeric>
#include <vector>

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
                        cweights[layer], bweights[layer], cgradients[layer], bgradients[layer],
                        order, alpha, learningRate, type);
        incoming_gradient = outgoing_gradient;
    }
    layerBackward(incoming_gradient, input, cweights[0], bweights[0], cgradients[0], bgradients[0],
                    order, alpha, learningRate, type);
}

/**
 * @brief Backpropagation for the mnn class (1D data) for batch backpropagation
 * @param expected The expected output vector.
 */
void mnn::backprop(const std::vector<std::vector<float>>& expected)
{
    int type = 3;
    std::vector<std::vector<float>> output_error(expected.size(), std::vector<float>(outSize, 0.0f));
    for(int i = 0; i < expected.size(); i++) {
        for(int j = 0; j < outSize; j++) {
            output_error[i][j] = activate[layers-1][i] - expected[i][j];
        }
    }
    std::vector<std::vector<float>> incoming_gradient = output_error;
    for(int layer = layers - 1; layer >= 1; layer--) {
        std::vector<std::vector<std::vector<float>>> origC(batchSize), origB(batchSize);
        std::vector<std::vector<float>> outgoing_gradient;
        for(int i = 0; i < batchSize; i++) {
            origC[i] = cweights[layer];
            origB[i] = bweights[layer];
            layerBackward(incoming_gradient[i], outgoing_gradient[i], activate[layer-1], 
                        origC[i], origB[i], cgradients[layer], bgradients[layer],
                        order, alpha, learningRate, type);
            incoming_gradient[i] = outgoing_gradient[i];
        }
        cweights[layer] = average(origC);
        bweights[layer] = average(origB);
    }
    std::vector<std::vector<std::vector<float>>> origC(batchSize), origB(batchSize);
    for(int i = 0; i < batchSize; i++) {
        origC[i] = cweights[0];
        origB[i] = bweights[0];
        layerBackward(incoming_gradient[i], input, origC[i], origB[i], cgradients[0], bgradients[0],
                    order, alpha, learningRate, type);
    }
    cweights[0] = average(origC);
    bweights[0] = average(origB);
}

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
                        cweights[layer], bweights[layer], cgradients[layer], bgradients[layer],
                        order, alpha, learningRate, type);
        incoming_gradient = outgoing_gradient;
    }
    layerBackward(incoming_gradient, reshape(softmax(flatten(input)), input.size(), input[0].size()),
                    cweights[0], bweights[0], cgradients[0], bgradients[0],
                    order, alpha, learningRate, type);
}


/**
 * @brief Backpropagation for the mnn2d class (2D data) for batch backpropagation
 * @param expected The expected output vector (after pooling).
 */
void mnn2d::backprop(const std::vector<std::vector<float>>& expected) {
    int type = 3;
    std::vector<std::vector<float>> output_error(expected.size(), std::vector<float>(expected[0].size(), 0.0f));
    for(int i = 0; i < expected.size(); i++) {
        for(int j = 0; j < expected[i].size(); j++)
            output_error[i][j] = output[i] - expected[i][j];
    }
    std::vector<std::vector<std::vector<float>>> incoming_gradient(batchSize);

    // Distribute output error to incoming gradient for mean pool
    for(int i = 0; i < batchSize; i++) {
        for(int j = 0; i < activate[layers-1].size(); i++) {
            incoming_gradient[i][j] = output_error[i];
        }
    }
    // Backpropagate the error
    for(int layer = layers - 1; layer >= 1; layer--) {
        std::vector<std::vector<std::vector<float>>> origC(batchSize), origB(batchSize);
        std::vector<std::vector<std::vector<float>>> outgoing_gradient;
        for(int i = 0; i < batchSize; i++) {
            origC[i] = cweights[layer];
            origB[i] = bweights[layer];
            layerBackward(incoming_gradient[i], outgoing_gradient[i], dotProds[layer-1], activate[layer-1],
                            cweights[layer], bweights[layer], cgradients[layer], bgradients[layer],
                            order, alpha, learningRate, type);
            incoming_gradient[i] = outgoing_gradient[i];
        }
        cweights[layer] = average(origC);
        bweights[layer] = average(origB);
    }
    std::vector<std::vector<std::vector<float>>> origC(batchSize), origB(batchSize);
    for(int i = 0; i < batchSize; i++) {
        origC[i] = cweights[0];
        origB[i] = bweights[0];
        layerBackward(incoming_gradient[i], reshape(softmax(flatten(input)), input.size(), input[0].size()),
                        cweights[0], bweights[0], cgradients[0], bgradients[0],
                        order, alpha, learningRate, type);
    }
    cweights[0] = average(origC);
    bweights[0] = average(origB);
}

#endif