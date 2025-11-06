#include "mnn.hpp"
#include <stdexcept>
#include <iostream>

/**
 * @brief train network on given dataset
 */
void mnn::train(const std::string &dataSetPath, int batchSize)
{
    // access all images from the file
    int totalFiles = 0;
    // train network using file one-by-one
    if (batchSize == 1) {
        std::vector<float> in;
        std::vector<float> exp;
        for(int i = 0; i < totalFiles; i++) {
            // make input and target
            #ifdef USE_CPU
                train(in, exp);
            #elif USE_CUDA
                cuTrain(in, exp);
            #elif USE_OPENCL
                clTrain(in, exp);
            #endif
        }
    }
    // train network using multiple files in batches
    else if (batchSize > 1) {
        // total batches
        int totalBatches = totalFiles / batchSize;
        std::vector<std::vector<float>> in;
        std::vector<std::vector<float>> exp;
        for(int i = 0; i < totalFiles; i += batchSize) {
            for(int j = 0; j < batchSize; j++) {
                // extract label from name
                // convert image to flat vector as input
            }
            #ifdef USE_CPU
                trainBatch(in, exp);
            #elif USE_CUDA
                cuTrainBatch(in, exp);
            #elif USE_OPENCL
                clTrainBatch(in, exp);
            #endif
        }
    }
    else {
        throw std::runtime_error("Invalid batch size: " + std::to_string(batchSize));
    }
}

/**
 * @brief train network on given dataset
 */
void mnn2d::train(const std::string &dataSetPath, int batchSize)
{
    // access all images from the file
    int totalFiles = 0;
    // train network using file one-by-one
    if (batchSize == 1) {
        for(int i = 0; i < totalFiles; i++) {
            std::vector<std::vector<float>> in;
            std::vector<float> exp;
            // make input and target
            #ifdef USE_CPU
                train(in, exp);
            #elif USE_CUDA
                cuTrain(in, exp);
            #elif USE_OPENCL
                clTrain(in, exp);
            #endif
        }
    }
    // train network using multiple files in batches
    else if (batchSize > 1) {
        // total batches
        int totalBatches = totalFiles / batchSize;
        std::vector<std::vector<std::vector<float>>> in;
        std::vector<std::vector<float>> exp;
        for(int i = 0; i < totalFiles; i += batchSize) {
            for(int j = 0; j < batchSize; j++) {
                // extract label from name
                // convert image to flat vector as input
            }
            // make input and target
            #ifdef USE_CPU
                trainBatch(in, exp);
            #elif USE_CUDA
                cuTrainBatch(in, exp);
            #elif USE_OPENCL
                clTrainBatch(in, exp);
            #endif
        }
    }
    else {
        throw std::runtime_error("Invalid batch size: " + std::to_string(batchSize));
    }
}