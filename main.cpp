#include <iostream>
#include <random>
#include "mnn.h"

int main() {
    std::cout << "THIS IS MONOMIAL NEURAL NETWORK  IMPLEMENTATION" << std::endl;
    std::string digitTrain      =   "D:\\train\\digits_mnist\\train";
    std::string digitTest       =   "D:\\train\\digits_mnist\\test";
    std::string fashionTrain    =   "D:\\train\\fashion_mnist\\train";
    std::string fashionTest     =   "D:\\train\\fashion_mnist\\test";
    std::string binFileAddress1 =   "D:\\monoNN\\weights1.bin";
    std::string binFileAddress2 =   "D:\\monoNN\\weights2.bin";

    int inSize = 784;
    int inh = 28, inw = 28;
    int outSize = 10;
    float order = 2.0f;
    std::vector<int> hidden_layers1 = { 784, 392, 196, 98, 49, outSize };
    std::vector<int> hidden_layers2 = { 28, 56, 112, 224, 224, 224, 112, 56, 28, outSize };

    try {
        std::cout << "---------------------MNN---------------------" << std::endl;

        mnn network(inSize, outSize, hidden_layers1, order, binFileAddress1);
        network.loadNetwork();
        network.batchSize = 10;
        network.train(digitTrain, network.batchSize);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred in main: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}