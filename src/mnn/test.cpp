#include "mnn.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>

/**
 * @brief test network on given dataset
 * @param dataSetPath path to test data set files
 * @param loss =
 */
void mnn::test(const std::string &dataSetPath, float loss)
{
    // access all data from dataset

    unsigned int totalInputs = 0;
    unsigned int correctPredictions = 0;
    unsigned int wrongPredictions = 0;

    std::vector<float> input;
    std::vector<float> target;

    float accLoss = 0.0f;       // accumulated loss
    for(int i = 0; i < totalInputs; i++) {
        #ifdef USE_CPU
            forprop(input);
        #elif USE_CUDA
            cuForprop(input);
        #elif USE_OPENCL
            clForprop(input);
        #endif
        if(maxIndex(output) == maxIndex(target)) {
            correctPredictions++;
        }
        else {
            wrongPredictions++;
        }
        accLoss += crossEntropy(output, target);
        if(i % 100 == 0) {
            std::cout << "Accumulated Loss till iteration " << i << ": " << accLoss << std::endl;
            std::cout << "Avgerage Loss till iteration " << i << ": " << accLoss/float(i) << std::endl;
            std::cout << "Correct predictions till iteration " << i << ": " << correctPredictions << std::endl;
            std::cout << "Wrong predictions till iteration " << i << ": " << wrongPredictions << std::endl;
        }
    }
    loss = accLoss / totalInputs;
}