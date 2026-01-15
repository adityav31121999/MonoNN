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
    std::string kmnist      =   path2Folder + "/kmnist";
    std::string fashion     =   path2Folder + "/fashion_mnist";

    int inSize1 = 784;
    int inh1 = 28, inw1 = 28;
    int outSize = 10;
    float order = 1.4f;
    bool useThreadOrBuffer = 1;
    bool isGreyOrRGB = 0;
    std::vector<int> width_mnn = {784, outSize};
    std::vector<int> width_mnn1 = {784, 448, 112, outSize};

    try {
        std::cout << "THIS IS MONOMIAL NEURAL NETWORK IMPLEMENTATION" << std::endl;

        std::cout << "----------------------MNN----------------------" << std::endl;
        mnn network1(inSize1, outSize, width_mnn, order, digit);
        network1.weightUpdateType = 0; network1.learningRate = LEARNING_MAX;
        network1.trainNtest(digit, isGreyOrRGB, useThreadOrBuffer, 0);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred in main: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}