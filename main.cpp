#include <iostream>
#include "mnn.h"

int main() {
    std::cout << "THIS IS MONOMIAL NEURAL NETWORK  IMPLEMENTATION" << std::endl;
    std::vector<int> hidden_layers1 = { 784, 392, 196, 98, 49, 10 };
    std::vector<int> hidden_layers2 = { 28, 56, 112, 224, 448, 224, 112, 56, 28, 10 };

    int inSize = 784, inh = 28, inw = 28;
    int outSize = 10;
    float order = 2.0f;
    std::string binFileAddress1 = "D:\\monoNN\\weights1.bin";
    std::string binFileAddress2 = "D:\\monoNN\\weights2.bin";

    mnn network(inSize, outSize, hidden_layers1, order, binFileAddress1);
    mnn2d net2d(inh, inw, outSize, hidden_layers2, order, binFileAddress2);

    return 0;
}