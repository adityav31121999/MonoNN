#include <vector>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include "mnn.hpp"
#include "mnn2d.hpp"

/**
 * @brief test network on given dataset
 * @param dataSetPath path to test data set folder.
 */
void mnn::test(const std::string &dataSetPath, bool useThreadOrBuffer)
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
        this->testPrg.testError = 0.0f;
        return;
    }

    unsigned int totalInputs = filePaths.size();
    unsigned int correctPredictions = 0;
    float accLoss = 0.0f; // accumulated loss

    std::cout << "\n--- Starting Test (mnn) ---" << std::endl;
    std::cout << "Found " << totalInputs << " files for testing." << std::endl;

    // Load previous progress to resume testing if applicable
    if (loadLastTestProgress(this->testPrg, this->path2test_progress)) {
        std::cout << "Successfully loaded test progress. Resuming testing." << std::endl;
        correctPredictions = this->testPrg.correctPredictions;
        accLoss = this->testPrg.testError * this->testPrg.testFilesProcessed; // Recalculate accumulated loss
    } else {
        std::cout << "No test progress file found or file is empty. Starting fresh test." << std::endl;
        this->testPrg = {}; // Reset test progress
    }
    std::cout << "Found " << totalInputs << " files for testing. Resuming from file index " << this->testPrg.testFilesProcessed << "." << std::endl;
    this->testPrg.totalTestFiles = totalInputs;

    for(size_t i = 0; i < totalInputs; ++i) {
        if (i < this->testPrg.testFilesProcessed) {
            continue; // Skip already processed files
        }
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
                      << " \t Accuracy: " << currentAccuracy * 100.0f << "%\t"
                      << " | Avg Loss: " << accLoss / (i + 1.0f) << std::endl;
            this->testPrg.testAccuracy = currentAccuracy;
            this->testPrg.testError = accLoss / (i + 1.0f);
            this->testPrg.testFilesProcessed = i + 1;
            this->testPrg.correctPredictions = correctPredictions;
            logTestProgressToCSV(this->testPrg, this->path2test_progress);
        }
    }

    this->testPrg.testError = (totalInputs > 0) ? (accLoss / totalInputs) : 0.0f;
    std::cout << "--- Test Finished (mnn) ---" << std::endl;
    std::cout << "Final Accuracy: " << ((float)correctPredictions / totalInputs) * 100.0f << "%" << std::endl;
    std::cout << "Final Average Loss: " << this->testPrg.testError << std::endl;
    std::cout << "Correct Predictions: " << correctPredictions << std::endl;
    std::cout << "Total Inputs: " << totalInputs << std::endl;
}

/**
 * @brief test network on given dataset
 * @param dataSetPath path to test data set folder.
 */
void mnn2d::test(const std::string &dataSetPath, bool useThreadOrBuffer)
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
        this->testPrg.testError = 0.0f;
        return;
    }

    unsigned int totalInputs = filePaths.size();
    unsigned int correctPredictions = 0;
    float accLoss = 0.0f;

    std::cout << "\n--- Starting Test (mnn2d) ---" << std::endl;
    std::cout << "Found " << totalInputs << " files for testing." << std::endl;

    // Load previous progress to resume testing if applicable
    if (loadLastTestProgress(this->testPrg, this->path2test_progress)) {
        std::cout << "Successfully loaded test progress. Resuming testing." << std::endl;
        correctPredictions = this->testPrg.correctPredictions;
        accLoss = this->testPrg.testError * this->testPrg.testFilesProcessed; // Recalculate accumulated loss
    } else {
        std::cout << "No test progress file found or file is empty. Starting fresh test." << std::endl;
        this->testPrg = {}; // Reset test progress
    }
    std::cout << "Found " << totalInputs << " files for testing. Resuming from file index " << this->testPrg.testFilesProcessed << "." << std::endl;
    this->testPrg.totalTestFiles = totalInputs;
    
    for(size_t i = 0; i < totalInputs; ++i) {
        if (i < this->testPrg.testFilesProcessed) {
            continue; // Skip already processed files
        }
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
                      << " | Avg Loss: " << accLoss / (i + 1.0f) << std::endl;
            this->testPrg.testAccuracy = currentAccuracy;
            this->testPrg.testError = accLoss / (i + 1.0f);
            this->testPrg.testFilesProcessed = i + 1;
            this->testPrg.correctPredictions = correctPredictions;
            logTestProgressToCSV(this->testPrg, this->path2test_progress);
        }
    }

    this->testPrg.testError = (totalInputs > 0) ? (accLoss / totalInputs) : 0.0f;
    std::cout << "--- Test Finished (mnn2d) ---" << std::endl;
    std::cout << "Final Accuracy: " << ((float)correctPredictions / totalInputs) * 100.0f << "%" << std::endl;
}