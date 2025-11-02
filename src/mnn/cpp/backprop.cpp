#ifdef USE_CPU
#include "mnn.hpp"
#include <numeric>
#include <vector>

/**
 * @brief Backpropagation for the mnn class (1D data).
 * @param expected The expected output vector.
 */
void mnn::backprop(std::vector<float>& expected) {
    int type = 3;
    this->target = expected;
    std::vector<float> output_error(outSize, 0.0f);
    for(int i = 0; i < outSize; i++) {
        output_error[i] = activate[layers-1][i] - expected[i];
    }
    std::vector<float> incoming_gradient = output_error;
    std::vector<float> outgoing_gradient;
    // Backpropagate the error
    for(int layer = layers - 1; layer >= 1; layer--) {
        layerBackward(incoming_gradient, outgoing_gradient, activate[layer-1], 
                        cweights[layer], bweights[layer], cgradients[layer], bgradients[layer],
                        order, alpha, learningRate, type);
        incoming_gradient = outgoing_gradient;
    }
    layerBackward(incoming_gradient, input,cweights[0], bweights[0], cgradients[0], bgradients[0],
                    order, alpha, learningRate, type);
}

/**
 * @brief Backpropagation for the mnn class (1D data) for batch backpropagation
 * @param expected The expected output vector.
 */
void mnn::backprop(std::vector<std::vector<float>>& expected)
{
    //
}

/**
 * @brief Backpropagation for the mnn2d class (2D data).
 * @param expected The expected output vector (after pooling).
 */
void mnn2d::backprop(std::vector<float>& expected) {
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
    layerBackward(incoming_gradient, input, reshape(softmax(flatten(input)), input.size(), input[0].size()),
                    cweights[0], bweights[0], cgradients[0], bgradients[0],
                    order, alpha, learningRate, type);
}


/**
 * @brief Backpropagation for the mnn2d class (2D data) for batch backpropagation
 * @param expected The expected output vector (after pooling).
 */
void mnn2d::backprop(std::vector<std::vector<float>>& expected) {
}

#endif