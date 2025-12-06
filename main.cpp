#include <iostream>
#include "mononn.h"

int main() {
    std::cout << "THIS IS MONOMIAL NEURAL NETWORK  IMPLEMENTATION" << std::endl;

    #ifdef __linux__
        std::string path2Folder = "/home/adi23444/code/train";
    #else
        std::string path2Folder = "D:/train";
    #endif

    std::string digitTrain      =   path2Folder + "/digits_mnist/train";
    std::string digitTest       =   path2Folder + "/digits_mnist/test";
    std::string fashionTrain    =   path2Folder + "/fashion_mnist/train";
    std::string fashionTest     =   path2Folder + "/fashion_mnist/test";
    std::string progressData1   =   path2Folder + "/progress/progress1.csv";   // mnn digits
    std::string progressData2   =   path2Folder + "/progress/progress2.csv";   // mnn fashion
    std::string progressData3   =   path2Folder + "/progress/progress3.csv";   // mnn2d digits
    std::string progressData4   =   path2Folder + "/progress/progress4.csv";   // mnn2d fashion
    std::string binFileAddress1 =   path2Folder + "/weightsMNNdigits.bin";
    std::string binFileAddress2 =   path2Folder + "/weightsMNNfashion.bin";
    std::string binFileAddress3 =   path2Folder + "/weightsMNN2Ddigits.bin";
    std::string binFileAddress4 =   path2Folder + "/weightsMNN2Dfashion.bin";

    int inSize = 784;
    int inh = 28, inw = 28;
    int outSize = 10;
    float order = 1.4f;
    bool batchMode = 1;
    std::vector<int> hidden_layers1 = { 784, 392, 196, 98, 49, outSize };
    std::vector<int> hidden_layers2 = { 28, 56, 112, 224, 224, 224, 112, 56, 28, outSize };

    try {
        std::cout << "----------------------MNN----------------------" << std::endl;
        mnn network1(inSize, outSize, hidden_layers1, order, binFileAddress1);
        // network1.initiateWeights(3);
        // network1.saveNetwork();
        network1.loadNetwork();
        network1.path2progress = progressData1;
        network1.mnnPrg.sessionSize = 50;
        network1.train(digitTrain, batchMode);

/*
        std::cout << "---------------------MNN2D---------------------" << std::endl;
        mnn2d network2(inh, inw, outSize, hidden_layers2, order, binFileAddress3);
        network2.loadNetwork();
        network2.path2progress = progressData3;
        network2.mnn2dPrg.sessionSize = 50;
        network2.train(digitTrain, batchMode);
*/
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred in main: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}