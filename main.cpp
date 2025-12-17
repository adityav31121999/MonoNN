#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include "mononn.h"

#define TRAIN_2D 0

int main() {
    try {
        std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error getting CWD: " << e.what() << std::endl;
    }

    #ifdef __linux__
        std::string path2Folder = "/home/aditya/code/train";
    #else
        std::string path2Folder = "D:/train";
    #endif

    std::string digit       =   path2Folder + "/digits_mnist";
    std::string fashion     =   path2Folder + "/digits_mnist";
    std::string cifar10     =   path2Folder + "/cifar10";

    int inSize = 784;
    int inh = 28, inw = 28;
    int outSize = 10;
    float order = 1.4f;
    bool useThreadOrBuffer = 0;
    std::vector<int> hidden_layers1 = { 784, 392, outSize };
    std::vector<int> hidden_layers2 = { 28, 56, 112, 112, 56, 28, outSize };
    std::vector<int> hidden_layers3 = { 1024, 512, 256, outSize };
    std::vector<int> hidden_layers4 = { 32, 64, 128, 256, 128, 64, 32, outSize };

    try {
        std::cout << "THIS IS MONOMIAL NEURAL NETWORK IMPLEMENTATION" << std::endl;

#if TRAIN_2D == 0
        std::cout << "----------------------MNN----------------------" << std::endl;
        mnn network1(inSize, outSize, hidden_layers1, order, digit);
        network1.weightUpdateType = 3;
        network1.trainPrg.sessionSize = 150;
        network1.fullDataSetTraining(digit, useThreadOrBuffer);
        // network1.test(digit, useThreadOrBuffer);
#else
        std::cout << "---------------------MNN2D---------------------" << std::endl;
        mnn2d network2(inh, inw, outSize, hidden_layers2, order, digit + "/mnn2d/weights.bin");
        network2.weightUpdateType = 3;
        network2.trainPrg.sessionSize = 50;
        network2.fullDataSetTraining(digit, useThreadOrBuffer);
        // network2.test(digit, useThreadOrBuffer);
#endif
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred in main: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}