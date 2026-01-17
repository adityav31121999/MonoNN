#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>
#include "mnn2d.hpp"

// for MNN2D

/**
 * @brief Post-training results.
 * @param dataSetPath path to complete data
 * @param isRGB if image is RGB 1, else 0 for grey image
 */
void mnn2d::postTrainRun(const std::string &dataSetPath, bool isRGB)
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
        std::cout << "\n--- Starting PostTrain Run on Training Set (mnn) ---" << std::endl;
        std::cout << "Order of Monomials: " << order << std::endl;
        std::sort(trainFilePaths.begin(), trainFilePaths.end());
        int totalTrainFiles = trainFilePaths.size();
        unsigned int correctPredictions = 0;
        int factor = totalTrainFiles / 100;
        confusion.assign(outSize, std::vector<int>(outSize, 0));
        allScores = {};
        allScores.totalSumOfRegression = 0.0f; allScores.totalSumOfError = 0.0f; allScores.totalSumOfSquares = 0.0f;
        allScores.r2 = 0.0f; allScores.sse = 0.0f; allScores.ssr = 0.0f; allScores.sst = 0.0f;
        confData = {};
        int count = 0;
        float accLoss = 0.0f;
        for(int i = 0; i < totalTrainFiles; i++) {
            const auto& filePath = trainFilePaths[i];
            std::vector<std::vector<float>> inputMat = image2matrix(filePath.string(), isRGB);
            std::string filename = filePath.stem().string();
            int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
            std::vector<float> exp(this->outSize, 0.0f);
            if (label < this->outSize) {
                exp[label] = 1.0f;
            }

            for(auto& row : inputMat) {
                for(auto& val : row) {
                    val /= 255.0f;
                }
            }

            #ifdef USE_CPU
                forprop(inputMat);
            #elif USE_CU
                cuForprop(inputMat);
            #elif USE_CL
                clForprop(inputMat);
            #endif

            if (maxIndex(output) == maxIndex(exp)) {
                correctPredictions++;
            }
            accLoss += crossEntropy(output, exp);
            getScore(output, exp, allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);
            if (label < confusion.size() && maxIndex(output) < confusion[0].size()) {
                confusion[label][maxIndex(output)]++;
            }

            count++;
            if(count % factor == 0) {
                std::cout << "Processed " << count << "/" << totalTrainFiles
                          << " files. Percent: " << static_cast<float>(count * 100) / totalTrainFiles << "%" << std::endl;
            }
        }

        float accuracy = (totalTrainFiles > 0) ? (static_cast<float>(correctPredictions) / totalTrainFiles) * 100.0f : 0.0f;
        float averageLoss = (totalTrainFiles > 0) ? accLoss / totalTrainFiles : 0.0f;

        std::cout << "PostTrain on Training Set Finished." << std::endl;
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
        std::vector<std::string> classes(outSize);
        for (int i = 0; i < outSize; ++i) {
            classes[i] = std::to_string(i);
        }
        printConfusionMatrix(confusion);
        printClassificationReport(confData, classes);
        computeStatsForCsv(cweights, bweights, weightStats);
        allScores.sse = allScores.totalSumOfError / totalTrainFiles;
        allScores.ssr = allScores.totalSumOfRegression / totalTrainFiles;
        allScores.sst = allScores.totalSumOfSquares / totalTrainFiles;
        allScores.r2 = allScores.ssr / allScores.sst;
        epochDataToCsv1(dataSetPath + "/mnn2d/post", 0, false, weightStats, confusion, confData, allScores, preTrainProgress, 1);
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
    else {
        std::cout << "\n--- Starting PostTrain Run on Test Set (mnn2d) ---" << std::endl;
        std::sort(testFilePaths.begin(), testFilePaths.end());
        int totalTestFiles = testFilePaths.size();
        unsigned int correctPredictions = 0;
        int factor = totalTestFiles / 100;
        confusion.assign(outSize, std::vector<int>(outSize, 0));
        allScores = {};
        allScores.totalSumOfRegression = 0.0f; allScores.totalSumOfError = 0.0f; allScores.totalSumOfSquares = 0.0f;
        allScores.r2 = 0.0f; allScores.sse = 0.0f; allScores.ssr = 0.0f; allScores.sst = 0.0f;
        int count = 0;
        float accLoss = 0.0f;
        for(int i = 0; i < totalTestFiles; i++) {
            const auto& filePath = testFilePaths[i];
            std::vector<std::vector<float>> inputMat = image2matrix(filePath.string(), isRGB);
            std::string filename = filePath.stem().string();
            int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
            std::vector<float> exp(this->outSize, 0.0f);
            if (label < this->outSize) {
                exp[label] = 1.0f;
            }

            for(auto& row : inputMat) {
                for(auto& val : row) {
                    val /= 255.0f;
                }
            }

            #ifdef USE_CPU
                forprop(inputMat);
            #elif USE_CU
                cuForprop(inputMat);
            #elif USE_CL
                clForprop(inputMat);
            #endif

            output = softmax(output, SOFTMAX_TEMP);
            if (maxIndex(output) == maxIndex(exp)) {
                correctPredictions++;
            }
            accLoss += crossEntropy(output, exp);
            getScore(output, exp, allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);
            if (label < confusion.size() && maxIndex(output) < confusion[0].size()) {
                confusion[label][maxIndex(output)]++;
            }

            count++;
            if(count % factor == 0) {
                std::cout << "Processed " << count << "/" << totalTestFiles
                          << " files. Percent: " << static_cast<float>(count * 100) / totalTestFiles << "%" << std::endl;
            }
        }

        float accuracy = (totalTestFiles > 0) ? (static_cast<float>(correctPredictions) / totalTestFiles) * 100.0f : 0.0f;
        float averageLoss = (totalTestFiles > 0) ? accLoss / totalTestFiles : 0.0f;

        std::cout << "PostTrain on Test Set Finished." << std::endl;
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
        std::vector<std::string> classes(outSize);
        for (int i = 0; i < outSize; ++i) {
            classes[i] = std::to_string(i);
        }
        printConfusionMatrix(confusion);
        printClassificationReport(confData, classes);
        allScores.sse = allScores.totalSumOfError / totalTestFiles;
        allScores.ssr = allScores.totalSumOfRegression / totalTestFiles;
        allScores.sst = allScores.totalSumOfSquares / totalTestFiles;
        allScores.r2 = allScores.ssr / allScores.sst;
        epochDataToCsv(dataSetPath + "/mnn2d/post", confusion, confData, allScores, preTestProgress, 1);
    }

    std::cout << "--- PostTraining Run Finished (mnn2d) ---" << std::endl;
}