#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include <random>
#include "mnn.hpp"
#include "mnn2d.hpp"

// for MNN

/**
 * @brief train network with full dataset passed every epoch
 * @param dataSetPath path to dataset folder
 * @param useThreadOrBuffer use threads in CPU and full buffer-based operation in CUDA and OpenCL
 */
void mnn::fullDataSetTraining(const std::string &dataSetPath, bool useThreadOrBuffer)
{
    // Access all image files from the dataset path
    std::vector<std::filesystem::path> filePaths;
    std::string trainPath = dataSetPath + "/train";

    try {
        for (const auto& entry : std::filesystem::directory_iterator(trainPath)) {
            if (entry.is_regular_file()) {
                filePaths.push_back(entry.path());
            }
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Failed to read dataset directory: " + std::string(e.what()));
    }

    if (filePaths.empty()) {
        std::cout << "Warning: No files found in dataset directory: " << trainPath << std::endl;
        return;
    }

    std::cout << "ALL HYPERPARAMETERS SET FOR TRAINING:" << std::endl;
    std::cout << "Order of Monomials: " << order << std::endl;
    std::cout << "Gradient Splitting Factor (ALPHA): " << ALPHA << std::endl;
    std::cout << "L1 Regularization Parameter (LAMBDA_L1): " << LAMBDA_L1 << std::endl;
    std::cout << "L2 Regularization Parameter (LAMBDA_L2): " << LAMBDA_L2 << std::endl;
    std::cout << "Dropout Rate (DROPOUT_RATE): " << DROPOUT_RATE << std::endl;
    std::cout << "Decay Rate (DECAY_RATE): " << DECAY_RATE << std::endl;
    std::cout << "Weight Decay Parameter (WEIGHT_DECAY): " << WEIGHT_DECAY << std::endl;
    std::cout << "Softmax Temperature (SOFTMAX_TEMP): " << SOFTMAX_TEMP << std::endl;
    this->learningRate = 0.001f;

    // Sort the dataset to ensure consistent order for resumable training
    std::sort(filePaths.begin(), filePaths.end());
    int totalFiles = filePaths.size();
    std::cout << "\n--- Starting Full Dataset Training (mnn) ---" << std::endl;

    // access training progress information from file address and use it to re-start training
    // from new session
    double previousTrainingTime = 0.0;
    int curPreds = 0;
    if (!loadLastProgress(this->trainPrg, this->path2progress)) {
        // Preserve session and batch size set before calling train
        unsigned int sessionSizeBackup = this->trainPrg.sessionSize;
        std::cout << "No progress file found or file is empty. Starting fresh training." << std::endl;
        this->trainPrg = {}; // Reset progress
        this->trainPrg.sessionSize = SESSION_SIZE;
        this->trainPrg.batchSize = 1;
        this->trainPrg.currentLearningRate = this->learningRate;
        this->trainPrg.sessionSize = sessionSizeBackup;
        curPreds = 0;
    }
    else {
        std::cout << "Successfully loaded progress." << std::endl;
        if(this->trainPrg.filesProcessed == 0)
            std::cout << "Fresh training starts!" << std::endl;
        else
            std::cout << "Starting from file index " << this->trainPrg.filesProcessed << " with epoch " << this->trainPrg.epoch << std::endl;
        this->learningRate = this->trainPrg.currentLearningRate;
        previousTrainingTime = this->trainPrg.timeTakenForTraining;
        std::cout << "Found " << totalFiles << " files for training. Resuming from file index " << this->trainPrg.filesProcessed << "." << std::endl;
        curPreds = this->trainPrg.trainingPredictions;
        epochs = this->trainPrg.epoch;
    }

    int fileCount = 0;
    int filesInCurrentSession = 0;
    // Update progress struct with file and batch info
    int sessionFiles = this->trainPrg.sessionSize; // * this->trainPrg.batchSize;
    std::cout << "Session Size (in batches/session): " << this->trainPrg.sessionSize << std::endl;
    std::cout << "Files in Single Session: " << sessionFiles << std::endl;
    std::cout << "learning rate: " << learningRate << std::endl;

    // Resize confusion matrix before use
    confusion.assign(outSize, std::vector<int>(outSize, 0));

    // start time count
    auto startTime = std::chrono::high_resolution_clock::now();

    while (1) {
        // Shuffle filePaths at the beginning of each epoch
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(filePaths.begin(), filePaths.end(), g);
        unsigned int correctPredictions = curPreds;
        for(const auto& filePath : filePaths) {
            // Skip files that have already been processed in previous sessions
            if (fileCount < this->trainPrg.filesProcessed) {
                fileCount++;
                continue;
            }

            // Convert image to a flat 1D vector
            std::vector<float> in = flatten(cvMat2vec(image2grey(filePath.string())));
            // Extract label from filename (e.g., "image_7.png" -> 7)
            std::string filename = filePath.stem().string();
            int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
            // Create one-hot encoded target vector
            std::vector<float> exp(this->outSize, 0.0f);
            if (label < this->outSize) {
                exp[label] = 1.0f;
            }
            // normalise input
            for(int i = 0; i < in.size(); i++) {
                in[i] /= 255;
            }
            input = in;
            target = exp;
            // backend selection
            #ifdef USE_CPU
                train1c(in, exp, useThreadOrBuffer);
            #elif USE_CU
                cuTrain1c(in, exp, useThreadOrBuffer);
            #elif USE_CL
                clTrain1c(in, exp, useThreadOrBuffer);
            #endif

            fileCount++;
            this->trainPrg.filesProcessed++;
            filesInCurrentSession++;
            if (label < confusion.size() && maxIndex(output) < confusion[0].size()) {
                confusion[label][maxIndex(output)]++;
            }
            if(maxIndex(output) == maxIndex(target)) correctPredictions++;
            this->trainPrg.accLoss += crossEntropy(output, target);
            this->allScores.totalSumOfError += static_cast<double>(sumOfSquareOfDiff(output, target));
            this->allScores.totalSumOfRegression += static_cast<double>(sumOfSquareOfDiff(target, mean(output)));
            this->allScores.totalSumOfSquares += static_cast<double>(sumOfSquareOfDiff(target, mean(target)));

            bool sessionEnd = 0;
            if ((sessionFiles > 0 && filesInCurrentSession == this->trainPrg.sessionSize) || fileCount == totalFiles) {
                auto endTime = std::chrono::high_resolution_clock::now();
                this->trainPrg.correctPredPercent = static_cast<float>(100 * correctPredictions) / fileCount;
                this->trainPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
                this->trainPrg.timeTakenForTraining = previousTrainingTime + this->trainPrg.timeForCurrentSession;
                this->learningRate = this->trainPrg.currentLearningRate;
                this->trainPrg.totalSessionsOfTraining++;
                this->trainPrg.totalCycleCount += sessionFiles;
                this->trainPrg.trainingPredictions = correctPredictions;
                sessionEnd = 1;
                startTime = std::chrono::high_resolution_clock::now();
                std::cout << "Epoch: " << this->trainPrg.epoch << "\tFiles Processed: " << fileCount << "/" << totalFiles
                          << " \tPredictions: " << correctPredictions
                          << " \tTraining Accuracy: " << this->trainPrg.correctPredPercent << "%"
                          << " \tLoss: " << this->trainPrg.accLoss / static_cast<float>(this->trainPrg.filesProcessed)
                          << std::endl;
            }

            // If a session size is defined and reached, stop training for this session
            if (sessionEnd == 1 || fileCount == totalFiles) {
                // computeStats(cweights, bweights, cgradients, bgradients, activate);
                if (logProgressToCSV(this->trainPrg, this->path2progress) != 1) {
                    throw std::runtime_error("Failed to log progress to CSV: " + this->path2progress);
                }
                serializeWeights(cweights, bweights, binFileAddress);
                filesInCurrentSession = 0; // Reset for the next session
                if (fileCount == totalFiles) {
                    std::cout << "All files processed. Ending training." << std::endl;
                    this->trainPrg.loss = this->trainPrg.accLoss / static_cast<float>(this->trainPrg.totalCycleCount);
                    bool notBatch = 0;
                    confData = {};
                    confData = confusionMatrixFunc(confusion);
                    allScores = {};
                    allScores.sse = allScores.totalSumOfError / totalFiles;
                    allScores.ssr = allScores.totalSumOfRegression / totalFiles;
                    allScores.sst = allScores.totalSumOfSquares / totalFiles;
                    allScores.r2 = allScores.ssr / allScores.sst;
                    epochDataToCsv(dataSetPath + "/mnn1d", epochs, notBatch,
                                    weightStats,
                                    confusion,
                                    confData, allScores, trainPrg);
                    break;
                }
            }
        }
        if(this->trainPrg.correctPredPercent >= 97.0f) {
            std::cout << "Training completed using minibatch of size " << BATCH_SIZE 
                      << "with accuracy of " << this->trainPrg.correctPredPercent << "%" << std::endl;
            break;
        }
        std::cout << "Training for next epoch: " << this->trainPrg.epoch << std::endl;
        this->trainPrg.epoch++;
        this->trainPrg.correctPredPercent = 0;
        this->trainPrg.accLoss = 0;
        this->trainPrg.filesProcessed = 0;
        this->trainPrg.timeForCurrentSession = 0;
        this->trainPrg.loss = 0;
        this->trainPrg.timeForCurrentSession = 0;
        logProgressToCSV(this->trainPrg, this->path2progress);
        allScores = {};
        confData = {};
        confusion.clear();
        confusion.assign(outSize, std::vector<int>(outSize, 0));
    }

    std::cout << "--- Training Finished (mnn) ---" << std::endl;
}

// for MNN2D

/**
 * @brief train network with full dataset passed every epoch
 * @param dataSetPath path to dataset folder
 * @param useThreadOrBuffer use threads in CPU and full buffer-based operation in CUDA and OpenCL
 */
void mnn2d::fullDataSetTraining(const std::string &dataSetPath, bool useThreadOrBuffer)
{
    // Access all image files from the dataset path
    std::vector<std::filesystem::path> filePaths;
    std::string trainPath = dataSetPath + "/train";
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(trainPath)) {
            if (entry.is_regular_file()) {
                filePaths.push_back(entry.path());
            }
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Failed to read dataset directory: " + std::string(e.what()));
    }

    if (filePaths.empty()) {
        std::cout << "Warning: No files found in dataset directory: " << trainPath << std::endl;
        return;
    }

    std::cout << "ALL HYPERPARAMETERS SET FOR TRAINING:" << std::endl;
    std::cout << "Order of Monomials: " << order << std::endl;
    std::cout << "Gradient Splitting Factor (ALPHA): " << ALPHA << std::endl;
    std::cout << "L1 Regularization Parameter (LAMBDA_L1): " << LAMBDA_L1 << std::endl;
    std::cout << "L2 Regularization Parameter (LAMBDA_L2): " << LAMBDA_L2 << std::endl;
    std::cout << "Dropout Rate (DROPOUT_RATE): " << DROPOUT_RATE << std::endl;
    std::cout << "Decay Rate (DECAY_RATE): " << DECAY_RATE << std::endl;
    std::cout << "Weight Decay Parameter (WEIGHT_DECAY): " << WEIGHT_DECAY << std::endl;
    std::cout << "Softmax Temperature (SOFTMAX_TEMP): " << SOFTMAX_TEMP << std::endl;
    this->learningRate = 0.001f;

    // Sort the dataset to ensure consistent order for resumable training
    std::sort(filePaths.begin(), filePaths.end());

    int totalFiles = filePaths.size();
    std::cout << "\n--- Starting Full Dataset Training (mnn) ---" << std::endl;

    // access training progress information from file address and use it to re-start training
    // from new session
    double previousTrainingTime = 0.0;
    int curPreds = 0;
    if (!loadLastProgress(this->trainPrg, this->path2progress)) {
        // Preserve session and batch size set before calling train
        unsigned int sessionSizeBackup = this->trainPrg.sessionSize;
        std::cout << "No progress file found or file is empty. Starting fresh training." << std::endl;
        this->trainPrg = {}; // Reset progress
        this->trainPrg.sessionSize = SESSION_SIZE;
        this->trainPrg.batchSize = 1;
        this->trainPrg.currentLearningRate = this->learningRate;
        this->trainPrg.sessionSize = sessionSizeBackup;
        curPreds = 0;
    }
    else {
        std::cout << "Successfully loaded progress." << std::endl;
        if(this->trainPrg.filesProcessed == 0)
            std::cout << "Fresh training starts!" << std::endl;
        else
            std::cout << "Starting from file index " << this->trainPrg.filesProcessed << " with epoch " << this->trainPrg.epoch << std::endl;
        this->learningRate = this->trainPrg.currentLearningRate;
        previousTrainingTime = this->trainPrg.timeTakenForTraining;
        std::cout << "Found " << totalFiles << " files for training. Resuming from file index " << this->trainPrg.filesProcessed << "." << std::endl;
        curPreds = this->trainPrg.trainingPredictions;
        epochs = this->trainPrg.epoch;
    }

    int fileCount = 0;
    int filesInCurrentSession = 0;
    // Update progress struct with file and batch info
    int sessionFiles = this->trainPrg.sessionSize; // * this->trainPrg.batchSize;
    std::cout << "Session Size (in batches/session): " << this->trainPrg.sessionSize << std::endl;
    std::cout << "Files in Single Session: " << sessionFiles << std::endl;
    std::cout << "learning rate: " << learningRate << std::endl;

    // Resize confusion matrix before use
    confusion.assign(outWidth, std::vector<int>(outWidth, 0));

    // start time count
    auto startTime = std::chrono::high_resolution_clock::now();

    while (1) {
        // Shuffle filePaths at the beginning of each epoch
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(filePaths.begin(), filePaths.end(), g);
        unsigned int correctPredictions = curPreds;
        for(const auto& filePath : filePaths) {
            // Skip files that have already been processed in previous sessions
            if (fileCount < this->trainPrg.filesProcessed) {
                fileCount++;
                continue;
            }

            // Convert image to a flat 1D vector
            std::vector<std::vector<float>> in = cvMat2vec(image2grey(filePath.string()));
            // Extract label from filename (e.g., "image_7.png" -> 7)
            std::string filename = filePath.stem().string();
            int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
            // Create one-hot encoded target vector
            std::vector<float> exp(this->outWidth, 0.0f);
            if (label < this->outWidth) {
                exp[label] = 1.0f;
            }
            for(int i = 0; i < in.size(); i++) {
                for(int j = 0; j < in[i].size(); j++) {
                    in[i][j] /= 255;
                }
            }
            target = exp;
            // backend selection
            #ifdef USE_CPU
                train1c(in, target, useThreadOrBuffer);
            #elif USE_CU
                cuTrain1c(in, target, useThreadOrBuffer);
            #elif USE_CL
                clTrain1c(in, target, useThreadOrBuffer);
            #endif

            fileCount++;
            this->trainPrg.filesProcessed++;
            filesInCurrentSession++;
            if (label < confusion.size() && maxIndex(output) < confusion[0].size()) {
                confusion[label][maxIndex(output)]++;
            }
            if(maxIndex(output) == maxIndex(target)) correctPredictions++;
            this->trainPrg.accLoss += crossEntropy(output, target);
            this->allScores.totalSumOfError += static_cast<double>(sumOfSquareOfDiff(output, target));
            this->allScores.totalSumOfRegression += static_cast<double>(sumOfSquareOfDiff(target, mean(output)));
            this->allScores.totalSumOfSquares += static_cast<double>(sumOfSquareOfDiff(target, mean(target)));

            bool sessionEnd = 0;
            if ((sessionFiles > 0 && filesInCurrentSession == this->trainPrg.sessionSize) || fileCount == totalFiles) {
                auto endTime = std::chrono::high_resolution_clock::now();
                this->trainPrg.correctPredPercent = static_cast<float>(100 * correctPredictions) / fileCount;
                this->trainPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
                this->trainPrg.timeTakenForTraining = previousTrainingTime + this->trainPrg.timeForCurrentSession;
                this->learningRate = this->trainPrg.currentLearningRate;
                this->trainPrg.totalSessionsOfTraining++;
                this->trainPrg.totalCycleCount += sessionFiles;
                this->trainPrg.trainingPredictions = correctPredictions;
                sessionEnd = 1;
                startTime = std::chrono::high_resolution_clock::now();
                std::cout << "Epoch: " << this->trainPrg.epoch << "\tFiles Processed: " << fileCount << "/" << totalFiles
                          << " \tPredictions: " << correctPredictions
                          << " \tTraining Accuracy: " << this->trainPrg.correctPredPercent << "%"
                          << " \tLoss: " << this->trainPrg.accLoss / static_cast<float>(this->trainPrg.filesProcessed)
                          << std::endl;
            }

            // If a session size is defined and reached, stop training for this session
            if (sessionEnd == 1 || fileCount == totalFiles) {
                // computeStats(cweights, bweights, cgradients, bgradients, activate);
                if (logProgressToCSV(this->trainPrg, this->path2progress) != 1) {
                    throw std::runtime_error("Failed to log progress to CSV: " + this->path2progress);
                }
                serializeWeights(cweights, bweights, binFileAddress);
                filesInCurrentSession = 0; // Reset for the next session
                if (fileCount == totalFiles) {
                    std::cout << "All files processed. Ending training." << std::endl;
                    this->trainPrg.loss = this->trainPrg.accLoss / static_cast<float>(this->trainPrg.totalCycleCount);
                    bool notBatch = 0;
                    confData = {};
                    confData = confusionMatrixFunc(confusion);
                    allScores = {};
                    allScores.sse = allScores.totalSumOfError / totalFiles;
                    allScores.ssr = allScores.totalSumOfRegression / totalFiles;
                    allScores.sst = allScores.totalSumOfSquares / totalFiles;
                    allScores.r2 = allScores.ssr / allScores.sst;
                    epochDataToCsv(dataSetPath + "/mnn1d", epochs, notBatch,
                                    weightStats,
                                    confusion,
                                    confData, allScores, trainPrg);
                    break;
                }
            }
        }
        if(this->trainPrg.correctPredPercent >= 97.0f) {
            std::cout << "Training completed using minibatch of size " << BATCH_SIZE 
                      << "with accuracy of " << this->trainPrg.correctPredPercent << "%" << std::endl;
            break;
        }
        std::cout << "Training for next epoch: " << this->trainPrg.epoch << std::endl;
        this->trainPrg.epoch++;
        this->trainPrg.correctPredPercent = 0;
        this->trainPrg.accLoss = 0;
        this->trainPrg.filesProcessed = 0;
        this->trainPrg.timeForCurrentSession = 0;
        this->trainPrg.loss = 0;
        this->trainPrg.timeForCurrentSession = 0;
        logProgressToCSV(this->trainPrg, this->path2progress);
        allScores = {};
        confData = {};
        confusion.clear();
        confusion.assign(outWidth, std::vector<int>(outWidth, 0));
    }

    std::cout << "--- Training Finished (mnn) ---" << std::endl;
}