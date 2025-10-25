#ifdef USE_CPU
#include "include/mnn.hpp"

void mnn::forprop(std::vector<float>& input)
{
    // use of operator * for vector and matrix multiplication
    // first layer
    layerForward(input, dotProds[0], getCLayer(1), getBLayer(1), order);
    activate[0] = sigmoid(dotProds[0]);

    // from 2nd to last
    for(int i = 1; i < layers; i++) {
        layerForward(activate[i-1], dotProds[i], getCLayer(i+1), getBLayer(i+1), order);
        activate[i] = sigmoid(dotProds[i]);
    }

    output = activate[layers - 1];
}

#endif