#include "mnn.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>

#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>

/**
 * @brief test network on given dataset
 * @param dataSetPath path to test data set folder
 * @param loss reference to a float to store the final average loss.
 */
void mnn::test(const std::string &dataSetPath, float& loss)
{
    // 1. Access all image files from the dataset path
    std::vector<std::filesystem::path> filePaths;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(dataSetPath)) {
            if (entry.is_regular_file()) {
                filePaths.push_back(entry.path());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Failed to read dataset directory: " + std::string(e.what()));
    }

    if (filePaths.empty()) {
        std::cout << "Warning: No files found in dataset directory: " << dataSetPath << std::endl;
        loss = 0.0f;
        return;
    }

    unsigned int totalInputs = filePaths.size();
    unsigned int correctPredictions = 0;
    float accLoss = 0.0f; // accumulated loss

    std::cout << "\n--- Starting Test (mnn) ---" << std::endl;
    std::cout << "Found " << totalInputs << " files for testing." << std::endl;

    for(size_t i = 0; i < totalInputs; ++i) {
        const auto& filePath = filePaths[i];

        // Prepare input
        std::vector<float> input = flatten(cvMat2vec(image2grey(filePath.string())));

        // Prepare target
        std::string filename = filePath.stem().string();
        int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
        std::vector<float> target(this->outSize, 0.0f);
        if (label < this->outSize) {
            target[label] = 1.0f;
        }

        // Perform forward propagation
        #ifdef USE_CPU
            forprop(input);
        #elif USE_CU
            cuForprop(input);
        #elif USE_CL
            clForprop(input);
        #endif

        // Check prediction
        if(maxIndex(this->output) == static_cast<size_t>(label)) {
            correctPredictions++;
        }

        accLoss += crossEntropy(this->output, target);

        if((i + 1) % 100 == 0 || (i + 1) == totalInputs) {
            float currentAccuracy = (float)correctPredictions / (i + 1);
            std::cout << "Processed " << i + 1 << "/" << totalInputs
                      << " | Accuracy: " << currentAccuracy * 100.0f << "%"
                      << " | Avg Loss: " << accLoss / (i + 1) << std::endl;
        }
    }

    loss = (totalInputs > 0) ? (accLoss / totalInputs) : 0.0f;
    std::cout << "--- Test Finished (mnn) ---" << std::endl;
    std::cout << "Final Accuracy: " << ((float)correctPredictions / totalInputs) * 100.0f << "%" << std::endl;
    std::cout << "Final Average Loss: " << loss << std::endl;
}

/**
 * @brief test network on given dataset
 * @param dataSetPath path to test data set folder
 * @param loss reference to a float to store the final average loss.
 */
void mnn2d::test(const std::string &dataSetPath, float& loss)
{
    std::vector<std::filesystem::path> filePaths;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(dataSetPath)) {
            if (entry.is_regular_file()) {
                filePaths.push_back(entry.path());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Failed to read dataset directory: " + std::string(e.what()));
    }

    if (filePaths.empty()) {
        std::cout << "Warning: No files found in dataset directory: " << dataSetPath << std::endl;
        loss = 0.0f;
        return;
    }

    unsigned int totalInputs = filePaths.size();
    unsigned int correctPredictions = 0;
    float accLoss = 0.0f;

    std::cout << "\n--- Starting Test (mnn2d) ---" << std::endl;
    std::cout << "Found " << totalInputs << " files for testing." << std::endl;

    for(size_t i = 0; i < totalInputs; ++i) {
        const auto& filePath = filePaths[i];
        std::vector<std::vector<float>> input = cvMat2vec(image2grey(filePath.string()));
        std::string filename = filePath.stem().string();
        int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
        std::vector<float> target(this->outWidth, 0.0f);
        if (label < this->outWidth) {
            target[label] = 1.0f;
        }

        #ifdef USE_CPU
            forprop(input);
        #elif USE_CU
            cuForprop(input);
        #elif USE_CL
            clForprop(input);
        #endif

        if(maxIndex(this->output) == static_cast<size_t>(label)) {
            correctPredictions++;
        }

        accLoss += crossEntropy(this->output, target);

        if((i + 1) % 100 == 0 || (i + 1) == totalInputs) {
            float currentAccuracy = (float)correctPredictions / (i + 1);
            std::cout << "Processed " << i + 1 << "/" << totalInputs
                      << " | Accuracy: " << currentAccuracy * 100.0f << "%"
                      << " | Avg Loss: " << accLoss / (i + 1) << std::endl;
        }
    }

    loss = (totalInputs > 0) ? (accLoss / totalInputs) : 0.0f;
    std::cout << "--- Test Finished (mnn2d) ---" << std::endl;
    std::cout << "Final Accuracy: " << ((float)correctPredictions / totalInputs) * 100.0f << "%" << std::endl;
    std::cout << "Final Average Loss: " << loss << std::endl;
}