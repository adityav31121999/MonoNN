#include <iostream>
#include "mnn.h"

int main() {
    std::cout << "THIS IS MONOMIAL NEURAL NETWORK  IMPLEMENTATION" << std::endl;
    std::vector<int> hidden_layers = { 784, 392, 196, 98, 49, 10 };

    int inSize = 784;
    int outSize = 10;
    int layers = hidden_layers.size();
    float order = 2.0f;
    std::string binFileAddress = "D:\\monoNN\\weights.bin";

    mnn network(inSize, outSize, hidden_layers, order, binFileAddress);

    return 0;
}