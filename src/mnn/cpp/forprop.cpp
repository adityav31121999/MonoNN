#ifdef USE_CPU
#include "mnn.hpp"
#include <vector>
#include <stdexcept>

/**
 * @brief forprop for monomial neural network with vector input
 * @param input input vector
 */
void mnn::forprop(const std::vector<float>& input)
{
    // std::vector<float> forPower;
    // use of operator * for vector and matrix multiplication
    // first layer
    layerForward(input, dotProds[0], getCWeights(1), getBWeights(1), order);
    activate[0] = sigmoid(dotProds[0]);

    // from 2nd to last
    for(int i = 1; i < layers; i++) {
        // forPower = power(activate[i-1], order); layerForward(forPower, dotProds[i], getCWeights(i+1), getBWeights(i+1));
        layerForward(activate[i-1], dotProds[i], getCWeights(i+1), getBWeights(i+1), order);
        activate[i] = sigmoid(dotProds[i]);
    }

    output = activate[layers - 1];
}


/**
 * @brief forprop for monomial neural network with matrix input
 * @param input input matrix
 */
void mnn2d::forprop(const std::vector<std::vector<float>>& input)
{
    // std::vector<float> forPower;
    // use of operator * for vector and matrix multiplication
    // first layer
    std::vector<std::vector<float>> activat(reshape(softmax(flatten(input)), input.size(), input[0].size()));
    layerForward(input, dotProds[0], getCWeights(1), getBWeights(1), order);
    activate[0] = reshape(softmax(flatten(dotProds[0])), dotProds[0].size(), dotProds[0][0].size());

    // from 2nd to last
    for(int i = 1; i < layers; i++) {
        // forPower = power(activate[i-1], order); layerForward(forPower, dotProds[i], getCWeights(i+1), getBWeights(i+1));
        layerForward(activate[i-1], dotProds[i], getCWeights(i+1), getBWeights(i+1), order);
        activate[i] = reshape(softmax(flatten(dotProds[i])), dotProds[i].size(), dotProds[i][0].size());
    }

    // apply mean pooling to the final activation layer to get output
    output = meanPool(activate[layers - 1]);
}



/**
 * @brief forprop for monomial neural network with vector input
 * @param input input vector
 */
void mnn::forprop(const std::vector<std::vector<float>>& input)
{
    // std::vector<float> forPower;
    // use of operator * for vector and matrix multiplication
    // first layer
    for(int i= 0; i < input.size(); i++) {
        layerForward(input[i], dotProdsBatch[0][i], getCWeights(1), getBWeights(1), order);
        activateBatch[0][i] = sigmoid(dotProdsBatch[0][i]);

        // from 2nd to last
        for(int j = 1; j < layers; j++) {
            // forPower = power(activate[i-1], order); layerForward(forPower, dotProds[i], getCWeights(i+1), getBWeights(i+1));
            layerForward(activateBatch[j-1][i], dotProdsBatch[j][i-1], getCWeights(j+1), getBWeights(j+1), order);
            activateBatch[j][i] = sigmoid(dotProdsBatch[j][i]);
        }
    }
    outputBatch = activateBatch[layers - 1];
}


/**
 * @brief forprop for monomial neural network with matrix input
 * @param input input matrix
 */
void mnn2d::forprop(const std::vector<std::vector<std::vector<float>>>& input)
{
    // std::vector<float> forPower;
    // use of operator * for vector and matrix multiplication
    // first layer
    for(int i = 0; i < input.size(); i++) {
        std::vector<std::vector<float>> activat(reshape(softmax(flatten(input[i])), input[i].size(), input[i][0].size()));
        layerForward(input[i], dotProdsBatch[0][i], getCWeights(1), getBWeights(1), order);
        activate[0] = reshape(softmax(flatten(dotProds[0])), dotProds[0].size(), dotProds[0][0].size());

        // from 2nd to last
        for(int j = 1; j < layers; j++) {
            // forPower = power(activate[i-1], order); layerForward(forPower, dotProds[i], getCWeights(i+1), getBWeights(i+1));
            layerForward(activateBatch[j-1][i], dotProdsBatch[j][i], getCWeights(j+1), getBWeights(j+1), order);
            activateBatch[j][i] = reshape(softmax(flatten(dotProdsBatch[j][i])), dotProdsBatch[j][i].size(), dotProdsBatch[j][i][0].size());
        }

        // apply mean pooling to the final activation layer to get output
        outputBatch[i] = meanPool(activateBatch[layers - 1][i]);
    }
}

#endif