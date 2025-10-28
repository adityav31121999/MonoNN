#ifdef USE_CPU
#include "mnn.hpp"

void mnn::forprop(std::vector<float>& input)
{
    // std::vector<float> forPower;
    // use of operator * for vector and matrix multiplication
    // first layer
    layerForward(input, dotProds[0], getCLayer(1), getBLayer(1), order);
    activate[0] = sigmoid(dotProds[0]);

    // from 2nd to last
    for(int i = 1; i < layers; i++) {
        // forPower = power(activate[i-1], order); layerForward(forPower, dotProds[i], getCLayer(i+1), getBLayer(i+1));
        layerForward(forPower, dotProds[i], getCLayer(i+1), getBLayer(i+1), order);
        activate[i] = sigmoid(dotProds[i]);
    }

    output = activate[layers - 1];
}

void mnn2d::forprop(std::vector<std::vector<float>>& input)
{
    // std::vector<float> forPower;
    // use of operator * for vector and matrix multiplication
    // first layer
    layerForward(input, dotProds[0], getCLayer(1), getBLayer(1), order);
    activate[0] = reshape(softmax(flatten(dotProds[0])), dotProds[0].size(), dotProds[0][0].size());

    // from 2nd to last
    for(int i = 1; i < layers; i++) {
        // forPower = power(activate[i-1], order); layerForward(forPower, dotProds[i], getCLayer(i+1), getBLayer(i+1));
        layerForward(activate[i-1], dotProds[i], getCLayer(i+1), getBLayer(i+1), order);
        activate[i] = reshape(softmax(flatten(dotProds[i])), dotProds[i].size(), dotProds[i][0].size());
    }

    // apply mean pooling to the final activation layer to get output
    output = meanPool(activate[layers - 1]);
}

#endif