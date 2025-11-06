#include <iostream>
#include <random>
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
    std::vector<float> input(inSize);
    std::vector<float> output(outSize);
    std::vector<float> target = { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < inSize; ++i) {
        input[i] = static_cast<float>(dis(gen));
    }
    network.train(input, target);
    return 0;
}

/*
    network.initiateWeights(3);
    std::vector<float> input(inSize);
    std::vector<float> output(outSize);
    std::vector<float> target = { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < inSize; ++i) {
        input[i] = static_cast<float>(dis(gen));
    }
    std::cout << "output: ";
    network.clForprop(input);
    for (int i = 0; i < outSize; ++i) {
        std::cout << network.output[i] << " ";
    }
    std::cout << std::endl;

    network.clBackprop(target);
    network.clForprop(input);
    for (int i = 0; i < outSize; ++i) {
        std::cout << network.output[i] << " ";
    }
*/