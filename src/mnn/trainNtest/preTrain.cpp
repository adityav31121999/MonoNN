#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>
#include "mnn.hpp"
#include "mnn2d.hpp"

// for MNN

/**
 * @brief pre-training results for weights(stats),set gradients (0 in stats) and prediction, perform with 
 *  both train + test files. This will be used in analysis of training and testing (epoch-wise).
 * @param dataSetPath path to complete dataset
 */
void mnn::preTrainRun(const std::string &dataSetPath)
{
    /*-----------------------
     --- Run on train set ---
     ------------------------*/
    std::vector<std::filesystem::path> trainFilePaths;
    std::string trainPath = dataSetPath + "/train";
    try {
        for (const auto& entry : std::filesystem::directory_iterator(trainPath)) {
            if (entry.is_regular_file()) {
                trainFilePaths.push_back(entry.path());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Failed to read train dataset directory: " + std::string(e.what()));
    }

    if (trainFilePaths.empty()) {
        std::cout << "Warning: No files found in train dataset directory: " << trainPath << std::endl;
    }
    else {
        std::cout << "\n--- Starting Pre-Train Run on Training Set (mnn) ---" << std::endl;
        std::sort(trainFilePaths.begin(), trainFilePaths.end());
        int totalTrainFiles = trainFilePaths.size();
        unsigned int correctPredictions = 0;
        float accLoss = 0.0f;

        confusion.assign(outSize, std::vector<int>(outSize, 0));
        allScores = {};

        for (const auto& filePath : trainFilePaths) {
            // Convert image to a flat 1D vector
            std::vector<float> in = flatten(cvMat2vec(image2grey(filePath.string())));
            for(auto& val : in) {
                val /= 255.0f;
            }

            // Extract label from filename (e.g., "image_7.png" -> 7)
            std::string filename = filePath.stem().string();
            int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));

            // Create one-hot encoded target vector
            std::vector<float> exp(this->outSize, 0.0f);
            if (label < this->outSize) {
                exp[label] = 1.0f;
            }

            #ifdef USE_CPU
                forprop(in);
            #elif USE_CU
                cuForprop(in);
            #elif USE_CL
                clForprop(in);
            #endif

            if (maxIndex(output) == maxIndex(exp)) {
                correctPredictions++;
            }
            accLoss += crossEntropy(output, exp);
            getScore(output, exp, allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);
            if (label < confusion.size() && maxIndex(output) < confusion[0].size()) {
                confusion[label][maxIndex(output)]++;
            }
        }

        float accuracy = (totalTrainFiles > 0) ? (static_cast<float>(correctPredictions) / totalTrainFiles) * 100.0f : 0.0f;
        float averageLoss = (totalTrainFiles > 0) ? accLoss / totalTrainFiles : 0.0f;

        std::cout << "Pre-Train on Training Set Finished." << std::endl;
        std::cout << "Total Files: " << totalTrainFiles << std::endl;
        std::cout << "Correct Predictions: " << correctPredictions << std::endl;
        std::cout << "Correct Predictions Percentage: " << accuracy << "%" << std::endl;
        std::cout << "Average Loss: " << averageLoss << std::endl;

        // Log results
        progress preTrainProgress = {};
        preTrainProgress.totalTrainFiles = totalTrainFiles;
        preTrainProgress.correctPredPercent = accuracy;
        preTrainProgress.loss = averageLoss;
        preTrainProgress.epoch = 0;

        confData = confusionMatrixFunc(confusion);
        computeStatsForCsv(cweights, bweights, cgradients, bgradients, activate, weightStats);
        allScores.sse = allScores.totalSumOfError / totalTrainFiles;
        allScores.ssr = allScores.totalSumOfRegression / totalTrainFiles;
        allScores.sst = allScores.totalSumOfSquares / totalTrainFiles;
        allScores.r2 = allScores.ssr / allScores.sst;
        epochDataToCsv(dataSetPath + "/mnn1d/pre", 0, false, weightStats, confusion, confData, allScores, preTrainProgress, false);
    }

    /*----------------------
     --- Run on test set ---
     -----------------------*/
    std::vector<std::filesystem::path> testFilePaths;
    std::string testPath = dataSetPath + "/test";
    try {
        for (const auto& entry : std::filesystem::directory_iterator(testPath)) {
            if (entry.is_regular_file()) {
                testFilePaths.push_back(entry.path());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Failed to read test dataset directory: " + std::string(e.what()));
    }

    if (testFilePaths.empty()) {
        std::cout << "Warning: No files found in dataset directory: " << testPath << std::endl;
        return;
    }

    std::cout << "\n--- Starting Pre-Train Run on Test Set (mnn) ---" << std::endl;
    std::sort(testFilePaths.begin(), testFilePaths.end());
    int totalTestFiles = testFilePaths.size();
    unsigned int correctPredictions = 0;
    float accLoss = 0.0f;

    confusion.assign(outSize, std::vector<int>(outSize, 0));
    allScores = {};

    for (const auto& filePath : testFilePaths) {
        // Convert image to a flat 1D vector
        std::vector<float> in = flatten(cvMat2vec(image2grey(filePath.string())));
        for(auto& val : in) {
            val /= 255.0f;
        }

        // Extract label from filename (e.g., "image_7.png" -> 7)
        std::string filename = filePath.stem().string();
        int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));

        // Create one-hot encoded target vector
        std::vector<float> exp(this->outSize, 0.0f);
        if (label < this->outSize) {
            exp[label] = 1.0f;
        }

        #ifdef USE_CPU
            forprop(in);
        #elif USE_CU
            cuForprop(in);
        #elif USE_CL
            clForprop(in);
        #endif

        if (maxIndex(output) == maxIndex(exp)) {
            correctPredictions++;
        }
        accLoss += crossEntropy(output, exp);
        getScore(output, exp, allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);
        if (label < confusion.size() && maxIndex(output) < confusion[0].size()) {
            confusion[label][maxIndex(output)]++;
        }
    }

    float accuracy = (totalTestFiles > 0) ? (static_cast<float>(correctPredictions) / totalTestFiles) * 100.0f : 0.0f;
    float averageLoss = (totalTestFiles > 0) ? accLoss / totalTestFiles : 0.0f;

    std::cout << "Pre-Train on Test Set Finished." << std::endl;
    std::cout << "Total Files: " << totalTestFiles << std::endl;
    std::cout << "Correct Predictions: " << correctPredictions << std::endl;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    std::cout << "Average Loss: " << averageLoss << std::endl;

    // Log results
    test_progress preTestProgress = {};
    preTestProgress.totalTestFiles = totalTestFiles;
    preTestProgress.testAccuracy = accuracy;
    preTestProgress.testError = averageLoss;
    preTestProgress.correctPredictions = correctPredictions;

    confData = confusionMatrixFunc(confusion);
    // Note: Gradients are zero in pre-training, so stats are only for weights.
    computeStatsForCsv(cweights, bweights, cgradients, bgradients, activate, weightStats);
    allScores.sse = allScores.totalSumOfError / totalTestFiles;
    allScores.ssr = allScores.totalSumOfRegression / totalTestFiles;
    allScores.sst = allScores.totalSumOfSquares / totalTestFiles;
    allScores.r2 = allScores.ssr / allScores.sst;
    epochDataToCsv(dataSetPath + "/mnn1d/pre", confusion, confData, allScores, preTestProgress, false);

    std::cout << "--- Pre-Training Run Finished (mnn) ---" << std::endl;
}

// for MNN2D

/**
 * @brief pre-training results for weights(stats),set gradients (0 in stats) and prediction, perform with 
 *  both train + test files. This will be used in analysis of training and testing (epoch-wise).
 * @param dataSetPath path to complete data 
 */
void mnn2d::preTrainRun(const std::string &dataSetPath)
{
    /*-----------------------
     --- Run on train set ---
     ------------------------*/
    std::vector<std::filesystem::path> trainFilePaths;
    std::string trainPath = dataSetPath + "/train";
    try {
        for (const auto& entry : std::filesystem::directory_iterator(trainPath)) {
            if (entry.is_regular_file()) {
                trainFilePaths.push_back(entry.path());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Failed to read train dataset directory: " + std::string(e.what()));
    }

    if (trainFilePaths.empty()) {
        std::cout << "Warning: No files found in train dataset directory: " << trainPath << std::endl;
    } else {
        std::cout << "\n--- Starting Pre-Train Run on Training Set (mnn2d) ---" << std::endl;
        std::sort(trainFilePaths.begin(), trainFilePaths.end());
        int totalTrainFiles = trainFilePaths.size();
        unsigned int correctPredictions = 0;
        float accLoss = 0.0f;

        confusion.assign(outWidth, std::vector<int>(outWidth, 0));
        allScores = {};

        for (const auto& filePath : trainFilePaths) {
            std::vector<std::vector<float>> in = cvMat2vec(image2grey(filePath.string()));
            for(auto& row : in) {
                for(auto& val : row) {
                    val /= 255.0f;
                }
            }

            std::string filename = filePath.stem().string();
            int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
            std::vector<float> exp(this->outWidth, 0.0f);
            if (label < this->outWidth) {
                exp[label] = 1.0f;
            }

            #ifdef USE_CPU
                forprop(in);
            #elif USE_CU
                cuForprop(in);
            #elif USE_CL
                clForprop(in);
            #endif

            if (maxIndex(output) == maxIndex(exp)) {
                correctPredictions++;
            }
            accLoss += crossEntropy(output, exp);
            getScore(output, exp, allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);
            if (label < confusion.size() && maxIndex(output) < confusion[0].size()) {
                confusion[label][maxIndex(output)]++;
            }
        }

        float accuracy = (totalTrainFiles > 0) ? (static_cast<float>(correctPredictions) / totalTrainFiles) * 100.0f : 0.0f;
        float averageLoss = (totalTrainFiles > 0) ? accLoss / totalTrainFiles : 0.0f;

        std::cout << "Pre-Train on Training Set Finished." << std::endl;
        std::cout << "Total Files: " << totalTrainFiles << std::endl;
        std::cout << "Correct Predictions: " << correctPredictions << std::endl;
        std::cout << "Correct Predictions Percentage: " << accuracy << "%" << std::endl;
        std::cout << "Average Loss: " << averageLoss << std::endl;

        // Log results
        progress preTrainProgress = {};
        preTrainProgress.totalTrainFiles = totalTrainFiles;
        preTrainProgress.correctPredPercent = accuracy;
        preTrainProgress.loss = averageLoss;
        preTrainProgress.epoch = 0;

        confData = confusionMatrixFunc(confusion);
        computeStatsForCsv(cweights, bweights, cgradients, bgradients, activate, weightStats);
        allScores.sse = allScores.totalSumOfError / totalTrainFiles;
        allScores.ssr = allScores.totalSumOfRegression / totalTrainFiles;
        allScores.sst = allScores.totalSumOfSquares / totalTrainFiles;
        allScores.r2 = allScores.ssr / allScores.sst;
        epochDataToCsv(dataSetPath + "/mnn2d/pre", 0, false, weightStats, confusion, confData, allScores, preTrainProgress, false);
    }

    /*----------------------
     --- Run on test set ---
     -----------------------*/
    std::vector<std::filesystem::path> testFilePaths;
    std::string testPath = dataSetPath + "/test";
    try {
        for (const auto& entry : std::filesystem::directory_iterator(testPath)) {
            if (entry.is_regular_file()) {
                testFilePaths.push_back(entry.path());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Failed to read test dataset directory: " + std::string(e.what()));
    }

    if (testFilePaths.empty()) {
        std::cout << "Warning: No files found in test dataset directory: " << testPath << std::endl;
        return;
    }

    std::cout << "\n--- Starting Pre-Train Run on Test Set (mnn2d) ---" << std::endl;
    std::sort(testFilePaths.begin(), testFilePaths.end());
    int totalTestFiles = testFilePaths.size();
    unsigned int correctPredictions = 0;
    float accLoss = 0.0f;

    confusion.assign(outWidth, std::vector<int>(outWidth, 0));
    allScores = {};

    for (const auto& filePath : testFilePaths) {
        std::vector<std::vector<float>> in = cvMat2vec(image2grey(filePath.string()));
        for(auto& row : in) {
            for(auto& val : row) {
                val /= 255.0f;
            }
        }

        std::string filename = filePath.stem().string();
        int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
        std::vector<float> exp(this->outWidth, 0.0f);
        if (label < this->outWidth) {
            exp[label] = 1.0f;
        }

        #ifdef USE_CPU
            forprop(in);
        #elif USE_CU
            cuForprop(in);
        #elif USE_CL
            clForprop(in);
        #endif

        if (maxIndex(output) == maxIndex(exp)) {
            correctPredictions++;
        }
        accLoss += crossEntropy(output, exp);
        getScore(output, exp, allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);
        if (label < confusion.size() && maxIndex(output) < confusion[0].size()) {
            confusion[label][maxIndex(output)]++;
        }
    }

    float accuracy = (totalTestFiles > 0) ? (static_cast<float>(correctPredictions) / totalTestFiles) * 100.0f : 0.0f;
    float averageLoss = (totalTestFiles > 0) ? accLoss / totalTestFiles : 0.0f;

    std::cout << "Pre-Train on Test Set Finished." << std::endl;
    std::cout << "Total Files: " << totalTestFiles << std::endl;
    std::cout << "Correct Predictions: " << correctPredictions << std::endl;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    std::cout << "Average Loss: " << averageLoss << std::endl;

    // Log results
    test_progress preTestProgress = {};
    preTestProgress.totalTestFiles = totalTestFiles;
    preTestProgress.testAccuracy = accuracy;
    preTestProgress.testError = averageLoss;
    preTestProgress.correctPredictions = correctPredictions;

    confData = confusionMatrixFunc(confusion);
    // Note: Gradients are zero in pre-training, so stats are only for weights.
    computeStatsForCsv(cweights, bweights, cgradients, bgradients, activate, weightStats);
    allScores.sse = allScores.totalSumOfError / totalTestFiles;
    allScores.ssr = allScores.totalSumOfRegression / totalTestFiles;
    allScores.sst = allScores.totalSumOfSquares / totalTestFiles;
    allScores.r2 = allScores.ssr / allScores.sst;
    epochDataToCsv(dataSetPath + "/mnn2d/pre", confusion, confData, allScores, preTestProgress, false);

    std::cout << "--- Pre-Training Run Finished (mnn2d) ---" << std::endl;
}