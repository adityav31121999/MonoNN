#include <vector>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include "mnn1d.hpp"
#include "mnn2d.hpp"

// for MNN

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
        testPrg.testError = 0.0f;
        return;
    }

    unsigned int totalInputs = filePaths.size();
    unsigned int correctPredictions = 0;
    float accLoss = 0.0f; // accumulated loss

    std::cout << "\n--- Starting Test (mnn) ---" << std::endl;
    std::cout << "Found " << totalInputs << " files for testing." << std::endl;
    testPrg.totalTestFiles = totalInputs;

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
        confusion[label][maxIndex(this->output)] += 1;
        // accumulate loss
        accLoss += crossEntropy(this->output, target);
        getScore(output, target, allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);

        if((i + 1) % 100 == 0 || (i + 1) == totalInputs) {
            float currentAccuracy = (float)correctPredictions / (i + 1);
            std::cout << "Processed " << i + 1 << "/" << totalInputs
                      << " \t Correct Prediction Percentage: " << currentAccuracy * 100.0f << "%\t"
                      << " | Avg Loss: " << accLoss / (i + 1.0f) << std::endl;
        }
    }

    testPrg.testError = (totalInputs > 0) ? (accLoss / totalInputs) : 0.0f;
    testPrg.correctPredictions = correctPredictions;
    testPrg.testAccuracy = static_cast<float>(correctPredictions * 100) / totalInputs;
    testPrg.totalTestFiles = totalInputs;
    logTestProgressToCSV(testPrg, path2test_progress);
    // evaluation
    allScores.sse = allScores.totalSumOfError / totalInputs;
    allScores.ssr = allScores.totalSumOfRegression / totalInputs;
    allScores.sst = allScores.totalSumOfSquares / totalInputs;
    allScores.r2 = allScores.ssr / allScores.sst;
    confData = {};
    confData = confusionMatrixFunc(confusion);
    epochDataToCsv(dataSetPath, confusion, confData, allScores, testPrg, true);
    path2test_progress = dataSetPath + "/mnn1d_test.csv";
    std::cout << "------ Final Result ------" << std::endl;
    std::cout << "Total Inputs: " << totalInputs << std::endl;
    std::cout << "Final Accuracy: " << ((float)correctPredictions / totalInputs) * 100.0f << "%" << std::endl;
    std::cout << "Final Average Loss: " << testPrg.testError << std::endl;
    std::cout << "Correct Predictions: " << correctPredictions << std::endl;
    std::cout << "--- Test Finished (mnn) ---" << std::endl;
}

// for MNN2D

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
        testPrg.testError = 0.0f;
        return;
    }

    unsigned int totalInputs = filePaths.size();
    unsigned int correctPredictions = 0;
    float accLoss = 0.0f;

    std::cout << "\n--- Starting Test (mnn2d) ---" << std::endl;
    std::cout << "Found " << totalInputs << " files for testing." << std::endl;
    testPrg.totalTestFiles = totalInputs;
    
    for(size_t i = 0; i < totalInputs; ++i) {
        const auto& filePath = filePaths[i];
        std::vector<std::vector<float>> input = cvMat2vec(image2grey(filePath.string()));
        std::string filename = filePath.stem().string();
        int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
        std::vector<float> target(this->outWidth, 0.0f);
        if (label < this->outWidth) {
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
        getScore(output, target, allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);
        // accumulate loss
        accLoss += crossEntropy(this->output, target);

        if((i + 1) % 200 == 0 || (i + 1) == totalInputs) {
            float currentAccuracy = (float)correctPredictions / (i + 1);
            std::cout << "Processed " << i + 1 << "/" << totalInputs
                      << " \t Accuracy: " << currentAccuracy * 100.0f << "%\t"
                      << " | Avg Loss: " << accLoss / (i + 1.0f) << std::endl;
        }
    }

    testPrg.testError = (totalInputs > 0) ? (accLoss / totalInputs) : 0.0f;
    testPrg.correctPredictions = correctPredictions;
    testPrg.testAccuracy = static_cast<float>(correctPredictions * 100) / totalInputs;
    testPrg.totalTestFiles = totalInputs;
    path2test_progress = dataSetPath + "/mnn1d_test.csv";
    logTestProgressToCSV(testPrg, path2test_progress);
    std::cout << "------- Final Result -------" << std::endl;
    std::cout << "Final Accuracy: " << ((float)correctPredictions / totalInputs) * 100.0f << "%" << std::endl;
    std::cout << "Final Average Loss: " << testPrg.testError << std::endl;
    std::cout << "Correct Predictions: " << correctPredictions << std::endl;
    std::cout << "Total Inputs: " << totalInputs << std::endl;
    std::cout << "--- Test Finished (mnn2d) ---" << std::endl;
}