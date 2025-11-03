#include "mnn.hpp"
#include <stdexcept>
#include <iostream>

void mnn::train(const std::string &dataSetPath, int batchSize)
{
    // access all images from the file
    int totalFiles = 0;
    // train network using file one-by-one
    if (batchSize == 1) {
        for(int i = 0; i < totalFiles; i++) {
            std::vector<float> input;
            std::vector<float> target;
            // extract label from name
            // convert image to flat vector as input
            #ifdef USE_CPU      // use C++ function
            #elif USE_CUDA      // use CUDA host-side function
            #elif USE_OPENCL    // use OpenCL host-side function
            #endif
        }
    }
    // train network using multiple files in batches
    else if (batchSize > 1) {
        // total batches
        int totalBatches = totalFiles / batchSize;
        std::vector<std::vector<float>> input;
        std::vector<std::vector<float>> target;
        for(int i = 0; i < totalFiles; i += batchSize) {
            for(int j = 0; j < batchSize; j++) {
                // extract label from name
                // convert image to flat vector as input
            }
            #ifdef USE_CPU      // use C++ function
            #elif USE_CUDA      // use CUDA host-side function
            #elif USE_OPENCL    // use OpenCL host-side function
            #endif
        }
    }
    else {
        throw std::runtime_error("Invalid batch size: " + std::to_string(batchSize));
    }
}

void mnn2d::train(const std::string &dataSetPath, int batchSize)
{
    // access all images from the file
    int totalFiles = 0;
    // train network using file one-by-one
    if (batchSize == 1) {
        for(int i = 0; i < totalFiles; i++) {
            std::vector<std::vector<float>> input;
            std::vector<float> target;
            // extract label from name
            // convert image to flat vector as input
            #ifdef USE_CPU      // use C++ function
            #elif USE_CUDA      // use CUDA host-side function
            #elif USE_OPENCL    // use OpenCL host-side function
            #endif
        }
    }
    // train network using multiple files in batches
    else if (batchSize > 1) {
        // total batches
        int totalBatches = totalFiles / batchSize;
        std::vector<std::vector<std::vector<float>>> input;
        std::vector<std::vector<float>> target;
        for(int i = 0; i < totalFiles; i += batchSize) {
            for(int j = 0; j < batchSize; j++) {
                // extract label from name
                // convert image to flat vector as input
            }
            #ifdef USE_CPU      // use C++ function
            #elif USE_CUDA      // use CUDA host-side function
            #elif USE_OPENCL    // use OpenCL host-side function
            #endif
        }
    }
    else {
        throw std::runtime_error("Invalid batch size: " + std::to_string(batchSize));
    }
}