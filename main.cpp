#include <iostream>
#include <random>
#include "mnn.h"

int main() {
    std::cout << "THIS IS MONOMIAL NEURAL NETWORK  IMPLEMENTATION" << std::endl;
    int inSize = 784;
    int inh = 28, inw = 28;
    int outSize = 10;
    float order = 2.0f;
    std::vector<int> hidden_layers1 = { 784, 392, 196, 98, 49, outSize };
    std::vector<int> hidden_layers2 = { 28, 56, 112, 224, 224, 224, 112, 56, 28, outSize };

    std::string binFileAddress1 = "D:\\monoNN\\weights1.bin";
    std::string binFileAddress2 = "D:\\monoNN\\weights2.bin";
    std::vector<float> target = { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0f, 1.0f);

    try {
        std::cout << "---------------------MNN---------------------" << std::endl;

        mnn network(inSize, outSize, hidden_layers1, order, binFileAddress1);
        network.initiateWeights(3);
        std::vector<float> input1(inSize, 0.0f);
        for (int i = 0; i < inSize; ++i) {
            input1[i] = static_cast<float>(dis(gen));
        }
        network.cuTrain(input1, target);

        std::cout << "--------------------MNN2D--------------------" << std::endl;

        std::vector<std::vector<float>> input2(inh, std::vector<float>(inw, 0.0f));
        mnn2d network2(inh, inw, outSize, hidden_layers2, order, binFileAddress2);
        network2.initiateWeights(3);
        for(int i = 0; i < inh; i++) {
            for(int j = 0; j < inw; j++) {
                input2[i][j] = static_cast<float>(dis(gen));
            }
        }
        network2.cuTrain(input2, target);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred in main: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}