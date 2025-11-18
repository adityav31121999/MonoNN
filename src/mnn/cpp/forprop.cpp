#ifdef USE_CPU
#include "mnn.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>

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
 * @brief forprop for monomial neural network with vector input
 * @param input input batch of vectors
 */
void mnn::forprop(const std::vector<std::vector<float>>& input)
{
    // std::vector<float> forPower;
    // use of operator * for vector and matrix multiplication
    // first layer
    for(int i= 0; i < input.size(); i++) {
        layerForward(input[i], dotBatch[0][i], cweights[0], bweights[0], order);
        actBatch[0][i] = sigmoid(dotBatch[0][i]);

        // from 2nd to last
        for(int j = 1; j < layers; j++) {
            // forPower = power(activate[i-1], order); layerForward(forPower, dotProds[i], cweights(i+1), bweights(i+1));
            layerForward(actBatch[j-1][i], dotBatch[j][i], cweights[j], bweights[j], order);
            actBatch[j][i] = sigmoid(dotBatch[j][i]);
        }
    }
    outputBatch = actBatch[layers - 1];
}


/**
 * @brief forprop for monomial neural network with matrix input
 * @param input input matrix
 */
void mnn2d::forprop(const std::vector<std::vector<float>>& input)
{
    // first layer
    std::vector<std::vector<float>> activat(reshape(softmax(flatten(input)), input.size(), input[0].size()));
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
 * @brief forprop for monomial neural network with matrix input
 * @param input input batch of matrix
 */
void mnn2d::forprop(const std::vector<std::vector<std::vector<float>>>& input)
{
    // std::vector<float> forPower;
    // use of operator * for vector and matrix multiplication
    // first layer
    for(int i = 0; i < input.size(); i++) {
        std::vector<std::vector<float>> activat(reshape(softmax(flatten(input[i])), input[i].size(), input[i][0].size()));
        layerForward(input[i], dotBatch[0][i], cweights[0], bweights[0], order);
        activate[0] = reshape(softmax(flatten(dotProds[0])), dotProds[0].size(), dotProds[0][0].size());

        // from 2nd to last
        for(int j = 1; j < layers; j++) {
            // forPower = power(activate[i-1], order); layerForward(forPower, dotProds[i], cweights(i+1), bweights(i+1));
            layerForward(actBatch[j-1][i], dotBatch[j][i], cweights[j], bweights[j], order);
            actBatch[j][i] = reshape(softmax(flatten(dotBatch[j][i])), dotBatch[j][i].size(), dotBatch[j][i][0].size());
        }

        // apply mean pooling to the final activation layer to get output
        outputBatch[i] = meanPool(actBatch[layers - 1][i]);
    }
}

#endif