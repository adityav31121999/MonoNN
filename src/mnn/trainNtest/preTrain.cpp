#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>
#include "mnn1d.hpp"
#include "mnn2d.hpp"

// for MNN

/**
 * @brief pre-training results for weights(stats),set gradients (0 in stats) and prediction, perform with 
 *  both train + test files. This will be used in analysis of training and testing (epoch-wise). Uses batch
 *  Forprop for fast execution.
 * @param dataSetPath path to complete dataset
 */
void mnn1d::preTrainRun(const std::string &dataSetPath)
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
        std::cout << "Batch Size: " << BATCH_SIZE << std::endl;
        std::cout << "Order of Monomials: " << order << std::endl;
        batchSize = BATCH_SIZE;
        this->inputBatch.resize(batchSize);
        this->outputBatch.resize(batchSize);
        this->targetBatch.resize(batchSize);
        this->dotBatch.resize(layers, std::vector<std::vector<float>>(batchSize));
        this->actBatch.resize(layers, std::vector<std::vector<float>>(batchSize));
        for(int j = 0; j < batchSize; j++) {
            this->inputBatch[j].resize(inSize, 0.0f);
            this->outputBatch[j].resize(outSize, 0.0f);
            this->targetBatch[j].resize(outSize, 0.0f);
        }
        for(int i = 0; i < layers; i++) {
            for(int j = 0; j < batchSize; j++) {
                this->dotBatch[i][j].resize(width[i]);
                this->actBatch[i][j].resize(width[i]);
            }
        }
        std::sort(trainFilePaths.begin(), trainFilePaths.end());
        int totalTrainFiles = trainFilePaths.size();
        unsigned int correctPredictions = 0;
        float accLoss = 0.0f;
        int factor = totalTrainFiles / 100;
        confusion.assign(outSize, std::vector<int>(outSize, 0));
        allScores = {};
        allScores.totalSumOfRegression = 0.0f; allScores.totalSumOfError = 0.0f; allScores.totalSumOfSquares = 0.0f;
        allScores.r2 = 0.0f; allScores.sse = 0.0f; allScores.ssr = 0.0f; allScores.sst = 0.0f;
        confData = {};
        int count = 0;
        for(int i = 0; i < totalTrainFiles; i += batchSize) {
            std::vector<std::vector<float>> inBatch;
            std::vector<std::vector<float>> expBatch;
            int currentBatchEnd = std::min<int>(i + batchSize, totalTrainFiles);
            // get image 
            for(int j = i; j < currentBatchEnd; ++j) {
                const auto& filePath = trainFilePaths[j];
                inBatch.push_back(flatten(cvMat2vec(image2grey(filePath.string()))));
                std::string filename = filePath.stem().string();
                int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
                std::vector<float> exp(this->outSize, 0.0f);
                if (label < this->outSize) {
                    exp[label] = 1.0f;
                }
                expBatch.push_back(exp);
            }
            for(int j = 0; j < inBatch.size(); j++) {
                for(size_t k = 0; k < inBatch[j].size(); k++) {
                    inBatch[j][k] /= 255.0f;
                }
            }

            inputBatch = inBatch;
            targetBatch = expBatch;
            // backend selection
            #ifdef USE_CPU
                forprop(inBatch);
            #elif USE_CU
                cuForprop(inBatch);
            #elif USE_CL
                clForprop(inBatch);
            #endif

            for (int j = 0; j < inBatch.size(); j++) {
                if(maxIndex(outputBatch[j]) == maxIndex(expBatch[j])) {
                    correctPredictions++;
                }
                // Update confusion matrix
                int label = maxIndex(expBatch[j]);
                if (label < confusion.size() && maxIndex(outputBatch[j]) < confusion[0].size()) {
                    confusion[label][maxIndex(outputBatch[j])]++;
                }
                trainPrg.accLoss += crossEntropy(outputBatch[j], expBatch[j]);
                getScore(outputBatch[j], expBatch[j], allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);
            }

            // for progress tracking
            count += batchSize;
            trainPrg.filesProcessed += batchSize;
            if(count % factor == 0) {
                std::cout << "Processed " << count << "/" << totalTrainFiles
                          << " files. Percent: " << static_cast<float>(count * 100) / totalTrainFiles << "%" << std::endl;
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
    else {
        std::cout << "\n--- Starting Pre-Train Run on Test Set (mnn) ---" << std::endl;
        std::sort(testFilePaths.begin(), testFilePaths.end());
        int totalTestFiles = testFilePaths.size();
        unsigned int correctPredictions = 0;
        float accLoss = 0.0f;
        int factor = totalTestFiles / 100;
        confusion.assign(outSize, std::vector<int>(outSize, 0));
        batchSize = BATCH_SIZE;
        this->inputBatch.resize(batchSize);
        this->outputBatch.resize(batchSize);
        this->targetBatch.resize(batchSize);
        this->dotBatch.resize(layers, std::vector<std::vector<float>>(batchSize));
        this->actBatch.resize(layers, std::vector<std::vector<float>>(batchSize));
        for(int j = 0; j < batchSize; j++) {
            this->inputBatch[j].resize(inSize, 0.0f);
            this->outputBatch[j].resize(outSize, 0.0f);
            this->targetBatch[j].resize(outSize, 0.0f);
        }
        for(int i = 0; i < layers; i++) {
            for(int j = 0; j < batchSize; j++) {
                this->dotBatch[i][j].resize(width[i]);
                this->actBatch[i][j].resize(width[i]);
            }
        }
        allScores = {};
        allScores.totalSumOfRegression = 0.0f; allScores.totalSumOfError = 0.0f; allScores.totalSumOfSquares = 0.0f;
        allScores.r2 = 0.0f; allScores.sse = 0.0f; allScores.ssr = 0.0f; allScores.sst = 0.0f;
        int count = 0;
        for(int i = 0; i < totalTestFiles; i += batchSize) {
            // Convert image to a flat 1D vector
            std::vector<std::vector<float>> inBatch;
            std::vector<std::vector<float>> expBatch;
            int currentBatchEnd = std::min<int>(i + batchSize, totalTestFiles);
            // get image 
            for(int j = i; j < currentBatchEnd; ++j) {
                const auto& filePath = trainFilePaths[j];
                inBatch.push_back(flatten(cvMat2vec(image2grey(filePath.string()))));
                std::string filename = filePath.stem().string();
                int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
                std::vector<float> exp(this->outSize, 0.0f);
                if (label < this->outSize) {
                    exp[label] = 1.0f;
                }
                expBatch.push_back(exp);
            }
            for(int j = 0; j < inBatch.size(); j++) {
                for(size_t k = 0; k < inBatch[j].size(); k++) {
                    inBatch[j][k] /= 255.0f;
                }
            }

            inputBatch = inBatch;
            targetBatch = expBatch;
            // backend selection
            #ifdef USE_CPU
                forprop(inBatch);
            #elif USE_CU
                cuForprop(inBatch);
            #elif USE_CL
                clForprop(inBatch);
            #endif

            for (int j = 0; j < inBatch.size(); j++) {
                if(maxIndex(outputBatch[j]) == maxIndex(expBatch[j])) {
                    correctPredictions++;
                }
                // Update confusion matrix
                int label = maxIndex(expBatch[j]);
                if (label < confusion.size() && maxIndex(outputBatch[j]) < confusion[0].size()) {
                    confusion[label][maxIndex(outputBatch[j])]++;
                }
                trainPrg.accLoss += crossEntropy(outputBatch[j], expBatch[j]);
                getScore(outputBatch[j], expBatch[j], allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);
            }

            // for progress tracking
            count += batchSize;
            trainPrg.filesProcessed += batchSize;
            if(count % factor == 0) {
                std::cout << "Processed " << count << "/" << totalTestFiles
                          << " files. Percent: " << static_cast<float>(count * 100) / totalTestFiles << "%" << std::endl;
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
        epochDataToCsv(dataSetPath + "/mnn1d/pre", confusion, confData, allScores, preTestProgress, false);
    }

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
        std::cout << "\n--- Starting Pre-Train Run on Training Set (mnn) ---" << std::endl;
        std::cout << "Batch Size: " << BATCH_SIZE << std::endl;
        std::cout << "Order of Monomials: " << order << std::endl;
        batchSize = BATCH_SIZE;
        this->inputBatch.resize(batchSize);
        this->outputBatch.resize(batchSize);
        this->targetBatch.resize(batchSize);
        this->dotBatch.resize(layers);
        this->actBatch.resize(layers);
        for(int j = 0; j < batchSize; j++) {
            this->inputBatch[j].resize(inHeight, std::vector<float>(inWidth, 0.0f));
            this->outputBatch[j].resize(outWidth, 0.0f);
            this->targetBatch[j].resize(outWidth, 0.0f);
        }
        for(int i = 0; i < layers; i++) {
            this->dotBatch[i].resize(batchSize);
            this->actBatch[i].resize(batchSize);
            for(int j = 0; j < batchSize; j++) {
                this->dotBatch[i][j].resize(inHeight, std::vector<float>(width[i], 0.0f));
                this->actBatch[i][j].resize(inHeight, std::vector<float>(width[i], 0.0f));
            }
        }
        std::sort(trainFilePaths.begin(), trainFilePaths.end());
        int totalTrainFiles = trainFilePaths.size();
        unsigned int correctPredictions = 0;
        float accLoss = 0.0f;
        int factor = totalTrainFiles / 100;
        confusion.assign(outWidth, std::vector<int>(outWidth, 0));
        allScores = {};
        allScores.totalSumOfRegression = 0.0f; allScores.totalSumOfError = 0.0f; allScores.totalSumOfSquares = 0.0f;
        allScores.r2 = 0.0f; allScores.sse = 0.0f; allScores.ssr = 0.0f; allScores.sst = 0.0f;
        confData = {};
        int count = 0;
        for(int i = 0; i < totalTrainFiles; i += batchSize) {
            std::vector<std::vector<std::vector<float>>> inBatch;
            std::vector<std::vector<float>> expBatch;
            int currentBatchEnd = std::min<int>(i + batchSize, totalTrainFiles);
            
            for(int j = i; j < currentBatchEnd; ++j) {
                const auto& filePath = trainFilePaths[j];
                inBatch.push_back(cvMat2vec(image2grey(filePath.string())));
                std::string filename = filePath.stem().string();
                int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
                std::vector<float> exp(this->outWidth, 0.0f);
                if (label < this->outWidth) {
                    exp[label] = 1.0f;
                }
                expBatch.push_back(exp);
            }

            for(auto& mat : inBatch) {
                for(auto& row : mat) {
                    for(auto& val : row) {
                        val /= 255.0f;
                    }
                }
            }

            inputBatch = inBatch;
            targetBatch = expBatch;

            #ifdef USE_CPU
                forprop(inBatch);
            #elif USE_CU
                cuForprop(inBatch);
            #elif USE_CL
                clForprop(inBatch);
            #endif

            for (int j = 0; j < inBatch.size(); j++) {
                if (maxIndex(outputBatch[j]) == maxIndex(expBatch[j])) {
                    correctPredictions++;
                }
                accLoss += crossEntropy(outputBatch[j], expBatch[j]);
                getScore(outputBatch[j], expBatch[j], allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);
                int label = maxIndex(expBatch[j]);
                if (label < confusion.size() && maxIndex(outputBatch[j]) < confusion[0].size()) {
                    confusion[label][maxIndex(outputBatch[j])]++;
                }
            }
            count += inBatch.size();
            if(count % factor == 0) {
                std::cout << "Processed " << count << "/" << totalTrainFiles
                          << " files. Percent: " << static_cast<float>(count * 100) / totalTrainFiles << "%" << std::endl;
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
        std::vector<std::string> classes(outWidth);
        for (int i = 0; i < outWidth; ++i) {
            classes[i] = std::to_string(i);
        }
        printConfusionMatrix(confusion);
        printClassificationReport(confData, classes);
        computeStatsForCsv(cweights, bweights, weightStats);
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
    else {
        std::cout << "\n--- Starting Pre-Train Run on Test Set (mnn2d) ---" << std::endl;
        std::sort(testFilePaths.begin(), testFilePaths.end());
        int totalTestFiles = testFilePaths.size();
        unsigned int correctPredictions = 0;
        float accLoss = 0.0f;
        int factor = totalTestFiles / 100;
        confusion.assign(outWidth, std::vector<int>(outWidth, 0));
        allScores = {};
        allScores.totalSumOfRegression = 0.0f; allScores.totalSumOfError = 0.0f; allScores.totalSumOfSquares = 0.0f;
        allScores.r2 = 0.0f; allScores.sse = 0.0f; allScores.ssr = 0.0f; allScores.sst = 0.0f;
        int count = 0;
        batchSize = BATCH_SIZE;
        this->inputBatch.resize(batchSize);
        this->outputBatch.resize(batchSize);
        this->targetBatch.resize(batchSize);
        this->dotBatch.resize(layers);
        this->actBatch.resize(layers);
        for(int j = 0; j < batchSize; j++) {
            this->inputBatch[j].resize(inHeight, std::vector<float>(inWidth, 0.0f));
            this->outputBatch[j].resize(outWidth, 0.0f);
            this->targetBatch[j].resize(outWidth, 0.0f);
        }
        for(int i = 0; i < layers; i++) {
            this->dotBatch[i].resize(batchSize);
            this->actBatch[i].resize(batchSize);
            for(int j = 0; j < batchSize; j++) {
                this->dotBatch[i][j].resize(inHeight, std::vector<float>(width[i], 0.0f));
                this->actBatch[i][j].resize(inHeight, std::vector<float>(width[i], 0.0f));
            }
        }
        for(int i = 0; i < totalTestFiles; i += batchSize) {
            std::vector<std::vector<std::vector<float>>> inBatch;
            std::vector<std::vector<float>> expBatch;
            int currentBatchEnd = std::min<int>(i + batchSize, totalTestFiles);
            
            for(int j = i; j < currentBatchEnd; ++j) {
                const auto& filePath = testFilePaths[j];
                inBatch.push_back(cvMat2vec(image2grey(filePath.string())));
                std::string filename = filePath.stem().string();
                int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
                std::vector<float> exp(this->outWidth, 0.0f);
                if (label < this->outWidth) {
                    exp[label] = 1.0f;
                }
                expBatch.push_back(exp);
            }

            for(auto& mat : inBatch) {
                for(auto& row : mat) {
                    for(auto& val : row) {
                        val /= 255.0f;
                    }
                }
            }

            inputBatch = inBatch;
            targetBatch = expBatch;

            #ifdef USE_CPU
                forprop(inBatch);
            #elif USE_CU
                cuForprop(inBatch);
            #elif USE_CL
                clForprop(inBatch);
            #endif

            for (int j = 0; j < inBatch.size(); j++) {
                if (maxIndex(outputBatch[j]) == maxIndex(expBatch[j])) {
                    correctPredictions++;
                }
                accLoss += crossEntropy(outputBatch[j], expBatch[j]);
                getScore(outputBatch[j], expBatch[j], allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);
                int label = maxIndex(expBatch[j]);
                if (label < confusion.size() && maxIndex(outputBatch[j]) < confusion[0].size()) {
                    confusion[label][maxIndex(outputBatch[j])]++;
                }
            }
            count += inBatch.size();
            if(count % factor == 0) {
                std::cout << "Processed " << count << "/" << totalTestFiles
                          << " files. Percent: " << static_cast<float>(count * 100) / totalTestFiles << "%" << std::endl;
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
        std::vector<std::string> classes(outWidth);
        for (int i = 0; i < outWidth; ++i) {
            classes[i] = std::to_string(i);
        }
        printConfusionMatrix(confusion);
        printClassificationReport(confData, classes);
        allScores.sse = allScores.totalSumOfError / totalTestFiles;
        allScores.ssr = allScores.totalSumOfRegression / totalTestFiles;
        allScores.sst = allScores.totalSumOfSquares / totalTestFiles;
        allScores.r2 = allScores.ssr / allScores.sst;
        epochDataToCsv(dataSetPath + "/mnn2d/pre", confusion, confData, allScores, preTestProgress, false);
    }

    std::cout << "--- Pre-Training Run Finished (mnn2d) ---" << std::endl;
}