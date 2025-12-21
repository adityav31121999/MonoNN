#ifdef USE_CPU
#include "mnn.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>

// forprop for mnn

/**
 * @brief forprop for monomial neural network with vector input
 * @param input input vector
 */
void mnn::forprop(const std::vector<float>& input)
{
    // first layer
    layerForward(input, dotProds[0], cweights[0], bweights[0], order);
    activate[0] = sigmoid(dotProds[0]);

    // from 2nd to last
    for(int i = 1; i < layers; i++) {
        layerForward(activate[i-1], dotProds[i], cweights[i], bweights[i], order);
        activate[i] = sigmoid(dotProds[i]);
    }

    output = activate[layers - 1];
}


/**
 * @brief batch forprop for monomial neural network with vector input
 * @param input batch input vectors
 */
void mnn::forprop(const std::vector<std::vector<float>>& input)
{
    if (input.empty())
        throw std::runtime_error("Input vector is empty");
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
                dotBatch[i][j].resize(width[i]);
                actBatch[i][j].resize(width[i]);
            }
        }
        // Reset dotBatch values to 0 as layerForwardBatch accumulates
        for (int j = 0; j < batchSize; ++j) {
            std::fill(dotBatch[i][j].begin(), dotBatch[i][j].end(), 0.0f);
        }
    }

    // first layer
    layerForwardBatch(input, dotBatch[0], cweights[0], bweights[0], order);
    for(int i = 0; i < batchSize; i++) {
        actBatch[0][i] = sigmoid(dotBatch[0][i]);
    }

    // from 2nd to last
    for(int j = 1; j < layers; j++) {
        layerForwardBatch(actBatch[j-1], dotBatch[j], cweights[j], bweights[j], order);
        for(int i = 0; i < batchSize; i++) {
            actBatch[j][i] = sigmoid(dotBatch[j][i]);
        }
    }

    outputBatch = actBatch[layers-1];
}

// forprop for mnn2d

/**
 * @brief forprop for monomial neural network with matrix input
 * @param input input matrix
 */
void mnn2d::forprop(const std::vector<std::vector<float>>& input)
{
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
        actBatch[0][i] = reshape(softmax(flatten(dotBatch[0][i])), dotBatch[0][i].size(), dotBatch[0][i][0].size());
    }

    // from 2nd to last
    for(int j = 1; j < layers; j++) {
        layerForwardBatch(actBatch[j-1], dotBatch[j], cweights[j], bweights[j], order);
        for(int i = 0; i < batchSize; i++) {
            actBatch[j][i] = reshape(softmax(flatten(dotBatch[j][i])), dotBatch[j][i].size(), dotBatch[j][i][0].size());
        }
    }

    // assign output batch
    for(int i = 0; i < batchSize; i++) {
        outputBatch[i] = meanPool(actBatch[layers-1][i]);
    }
}

#endif