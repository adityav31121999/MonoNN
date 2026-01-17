#ifdef USE_CPU
#include "mnn1d.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>

/**
 * @brief forprop for monomial neural network with matrix input
 * @param input input matrix
 */
void mnn2d::forprop(const std::vector<std::vector<float>>& input)
{
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
}


/**
 * @brief batch forprop for monomial neural network with matrix input
 * @param input input batch of matrix
 */
void mnn2d::forprop(const std::vector<std::vector<std::vector<float>>>& input)
{
    if (input.empty()) return;
    this->batchSize = input.size();

    if (dotBatch.size() != layers) {
        dotBatch.resize(layers);
        actBatch.resize(layers);
    }
    for (int i = 0; i < layers; ++i) {
        if (dotBatch[i].size() != batchSize) {
            dotBatch[i].resize(batchSize);
            actBatch[i].resize(batchSize);
            for (int j = 0; j < batchSize; ++j) {
                dotBatch[i][j].resize(inHeight, std::vector<float>(width[i], 0.0f));
                actBatch[i][j].resize(inHeight, std::vector<float>(width[i]));
            }
        }
        // Reset dotBatch values to 0
        for (int j = 0; j < batchSize; ++j) {
            for (int r = 0; r < inHeight; ++r) {
                std::fill(dotBatch[i][j][r].begin(), dotBatch[i][j][r].end(), 0.0f);
            }
        }
    }

    // first layer
    layerForwardBatch(input, dotBatch[0], cweights[0], bweights[0], order);
    for(int i = 0; i < batchSize; i++) {
        actBatch[0][i] = relu(dotBatch[0][i]);
    }

    // from 2nd to last
    for(int j = 1; j < layers; j++) {
        layerForwardBatch(actBatch[j-1], dotBatch[j], cweights[j], bweights[j], order);
        for(int i = 0; i < batchSize; i++) {
            actBatch[j][i] = relu(dotBatch[j][i]);
        }
    }

    // assign output batch
    for(int i = 0; i < batchSize; i++) {
        outputBatch[i] = softmax(meanPool(actBatch[layers-1][i]));
    }
}

#endif