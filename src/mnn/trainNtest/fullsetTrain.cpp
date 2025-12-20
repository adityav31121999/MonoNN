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

    std::cout << "ALL HYPERPARAMETERS SET FOR TRAINING ARE:" << std::endl;
    std::cout << "> Order of Monomials: " << order << std::endl;
    std::cout << "> Gradient Splitting Factor (ALPHA): " << ALPHA << std::endl;
    std::cout << "> L1 Regularization Parameter (LAMBDA_L1): " << LAMBDA_L1 << std::endl;
    std::cout << "> L2 Regularization Parameter (LAMBDA_L2): " << LAMBDA_L2 << std::endl;
    std::cout << "> Dropout Rate (DROPOUT_RATE): " << DROPOUT_RATE << std::endl;
    std::cout << "> Decay Rate (DECAY_RATE): " << DECAY_RATE << std::endl;
    std::cout << "> Weight Decay Parameter (WEIGHT_DECAY): " << WEIGHT_DECAY << std::endl;
    std::cout << "> Softmax Temperature (SOFTMAX_TEMP): " << SOFTMAX_TEMP << std::endl;
    this->learningRate = 0.001f;

    // Sort the dataset to ensure consistent order for resumable training
    std::sort(filePaths.begin(), filePaths.end());
    int totalFiles = filePaths.size();
    std::cout << "\n--- Starting Full Dataset Training (mnn) ---" << std::endl;
    
    if(useThreadOrBuffer == 1) {
        #if defined(USE_CU) || defined(USE_CL)
            std::cout << "> Using Common Buffers for single cycle of training" << std::endl;
        #else
            std::cout << "> Using threads for single cycle of training" << std::endl;
        #endif
    }
    else {
        std::cout << "> Using standalone distinct functions for single cycle of training" << std::endl;
    }

    // access training progress information from file address and use it to re-start training
    // from new session
    double previousTrainingTime = 0.0;
    int curPreds = 0;
    if (!loadLastProgress(trainPrg, this->path2progress)) {
        std::cout << "No progress file found or file is empty. Starting fresh training." << std::endl;
        trainPrg = {}; // Reset progress
        trainPrg.sessionSize = SESSION_SIZE;
        trainPrg.batchSize = 1;
        trainPrg.currentLearningRate = this->learningRate;
        curPreds = 0;
    }
    else {
        std::cout << "Successfully loaded progress." << std::endl;
        if(trainPrg.filesProcessed == 0)
            std::cout << "Fresh training starts!" << std::endl;
        else
            std::cout << "Starting from file index " << trainPrg.filesProcessed << " with epoch " << trainPrg.epoch << std::endl;
        this->learningRate = trainPrg.currentLearningRate;
        previousTrainingTime = trainPrg.timeTakenForTraining;
        std::cout << "Found " << totalFiles << " files for training. Resuming from file index " << trainPrg.filesProcessed << "." << std::endl;
        curPreds = trainPrg.trainingPredictions;
        epochs = trainPrg.epoch;
    }

    int fileCount = 0;
    int filesInCurrentSession = 0;
    // Update progress struct with file and batch info
    int sessionFiles = trainPrg.sessionSize; // * trainPrg.batchSize;
    std::cout << "Session Size (in batches/session): " << trainPrg.sessionSize << std::endl;
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
            if (fileCount < trainPrg.filesProcessed) {
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
            trainPrg.filesProcessed++;
            filesInCurrentSession++;
            if (label < confusion.size() && maxIndex(output) < confusion[0].size()) {
                confusion[label][maxIndex(output)]++;
            }
            if(maxIndex(output) == maxIndex(target)) correctPredictions++;
            trainPrg.accLoss += crossEntropy(output, target);
            getScore(output, target, allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);

            bool sessionEnd = 0;
            if ((sessionFiles > 0 && filesInCurrentSession == trainPrg.sessionSize) || fileCount == totalFiles) {
                auto endTime = std::chrono::high_resolution_clock::now();
                // progress file
                trainPrg.correctPredPercent = static_cast<float>(100 * correctPredictions) / fileCount;
                trainPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
                trainPrg.timeTakenForTraining = previousTrainingTime + trainPrg.timeForCurrentSession;
                this->learningRate = trainPrg.currentLearningRate;
                trainPrg.sessionCount++;
                trainPrg.totalCycleCount += sessionFiles;
                trainPrg.trainingPredictions = correctPredictions;
                // confusion matrix
                confData = {};
                confData = confusionMatrixFunc(confusion);
                // stats
                computeStatsForCsv(cweights, bweights, cgradients, bgradients, activate, weightStats);
                // scores
                allScores.sse = allScores.totalSumOfError / trainPrg.filesProcessed;
                allScores.ssr = allScores.totalSumOfRegression / trainPrg.filesProcessed;
                allScores.sst = allScores.totalSumOfSquares / trainPrg.filesProcessed;
                allScores.r2 = allScores.ssr / allScores.sst;
                // save session data
                sessionDataToCsv(path2SessionDir, trainPrg.epoch, trainPrg.sessionCount, false,
                                    weightStats, confusion, confData, allScores, trainPrg);
                sessionEnd = 1;
                startTime = std::chrono::high_resolution_clock::now();
                std::cout << "Epoch: " << trainPrg.epoch << "\tFiles Processed: " << fileCount << "/" << totalFiles
                          << " \tPredictions: " << correctPredictions
                          << " \tTraining Accuracy: " << trainPrg.correctPredPercent << "%"
                          << " \tLoss: " << trainPrg.accLoss / static_cast<float>(trainPrg.filesProcessed)
                          << std::endl;
            }

            // If a session size is defined and reached, serialise weights
            if (sessionEnd == 1 || fileCount == totalFiles) {
                // computeStats(cweights, bweights, cgradients, bgradients, activate);
                if (logProgressToCSV(trainPrg, this->path2progress) != 1) {
                    throw std::runtime_error("Failed to log progress to CSV: " + this->path2progress);
                }
                serializeWeights(cweights, bweights, binFileAddress);
                filesInCurrentSession = 0;
                if (fileCount == totalFiles) {
                    std::cout << "All files processed. Ending training." << std::endl;
                    trainPrg.loss = trainPrg.accLoss / static_cast<float>(trainPrg.totalCycleCount);
                    bool notBatch = 0;
                    confData = {};
                    confData = confusionMatrixFunc(confusion);
                    allScores = {};
                    allScores.sse = allScores.totalSumOfError / totalFiles;
                    allScores.ssr = allScores.totalSumOfRegression / totalFiles;
                    allScores.sst = allScores.totalSumOfSquares / totalFiles;
                    allScores.r2 = allScores.ssr / allScores.sst;
                    epochDataToCsv(dataSetPath + "/mnn1d", trainPrg.epoch, notBatch, weightStats, confusion, confData, allScores,
                                    trainPrg, true);
                    break;
                }
            }
        }

        if(trainPrg.correctPredPercent >= 97.0f) {
            std::cout << "Training completed using minibatch of size " << BATCH_SIZE 
                      << "with accuracy of " << trainPrg.correctPredPercent << "%" << std::endl;
            break;
        }

        std::cout << "Training for next epoch: " << trainPrg.epoch + 1 << std::endl;
        trainPrg.epoch++;
        trainPrg.sessionSize = SESSION_SIZE;
        trainPrg.filesProcessed = 0;
        trainPrg.batchSize = 1;
        trainPrg.trainingPredictions = 0;
        trainPrg.currentLearningRate = this->learningRate;
        trainPrg.loss = 0;
        trainPrg.accLoss = 0;
        trainPrg.correctPredPercent = 0;
        trainPrg.sessionCount= 0;
        trainPrg.timeForCurrentSession = 0;
        logProgressToCSV(trainPrg, this->path2progress);

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

    std::cout << "ALL HYPERPARAMETERS SET FOR TRAINING ARE:" << std::endl;
    std::cout << "> Order of Monomials: " << order << std::endl;
    std::cout << "> Gradient Splitting Factor (ALPHA): " << ALPHA << std::endl;
    std::cout << "> L1 Regularization Parameter (LAMBDA_L1): " << LAMBDA_L1 << std::endl;
    std::cout << "> L2 Regularization Parameter (LAMBDA_L2): " << LAMBDA_L2 << std::endl;
    std::cout << "> Dropout Rate (DROPOUT_RATE): " << DROPOUT_RATE << std::endl;
    std::cout << "> Decay Rate (DECAY_RATE): " << DECAY_RATE << std::endl;
    std::cout << "> Weight Decay Parameter (WEIGHT_DECAY): " << WEIGHT_DECAY << std::endl;
    std::cout << "> Softmax Temperature (SOFTMAX_TEMP): " << SOFTMAX_TEMP << std::endl;
    this->learningRate = 0.001f;

    // Sort the dataset to ensure consistent order for resumable training
    std::sort(filePaths.begin(), filePaths.end());
    int totalFiles = filePaths.size();
    std::cout << "\n--- Starting Full Dataset Training (mnn) ---" << std::endl;
    
    if(useThreadOrBuffer == 1) {
        #if defined(USE_CU) || defined(USE_CL)
            std::cout << "> Using Common Buffers for single cycle of training" << std::endl;
        #else
            std::cout << "> Using threads for single cycle of training" << std::endl;
        #endif
    }
    else {
        std::cout << "> Using standalone distinct functions for single cycle of training" << std::endl;
    }

    // access training progress information from file address and use it to re-start training
    // from new session
    double previousTrainingTime = 0.0;
    int curPreds = 0;
    if (!loadLastProgress(trainPrg, this->path2progress)) {
        // Preserve session and batch size set before calling train
        unsigned int sessionSizeBackup = trainPrg.sessionSize;
        std::cout << "No progress file found or file is empty. Starting fresh training." << std::endl;
        trainPrg = {}; // Reset progress
        trainPrg.sessionSize = SESSION_SIZE;
        trainPrg.batchSize = 1;
        trainPrg.currentLearningRate = this->learningRate;
        trainPrg.sessionSize = sessionSizeBackup;
        curPreds = 0;
    }
    else {
        std::cout << "Successfully loaded progress." << std::endl;
        if(trainPrg.filesProcessed == 0)
            std::cout << "Fresh training starts!" << std::endl;
        else
            std::cout << "Starting from file index " << trainPrg.filesProcessed << " with epoch " << trainPrg.epoch << std::endl;
        this->learningRate = trainPrg.currentLearningRate;
        previousTrainingTime = trainPrg.timeTakenForTraining;
        std::cout << "Found " << totalFiles << " files for training. Resuming from file index " << trainPrg.filesProcessed << "." << std::endl;
        curPreds = trainPrg.trainingPredictions;
        epochs = trainPrg.epoch;
    }

    int fileCount = 0;
    int filesInCurrentSession = 0;
    // Update progress struct with file and batch info
    int sessionFiles = trainPrg.sessionSize; // * trainPrg.batchSize;
    std::cout << "Session Size (in batches/session): " << trainPrg.sessionSize << std::endl;
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
            if (fileCount < trainPrg.filesProcessed) {
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
            trainPrg.filesProcessed++;
            filesInCurrentSession++;
            if (label < confusion.size() && maxIndex(output) < confusion[0].size()) {
                confusion[label][maxIndex(output)]++;
            }
            if(maxIndex(output) == maxIndex(target)) correctPredictions++;
            trainPrg.accLoss += crossEntropy(output, target);
            getScore(output, target, allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);

            bool sessionEnd = 0;
            if ((sessionFiles > 0 && filesInCurrentSession == trainPrg.sessionSize) || fileCount == totalFiles) {
                auto endTime = std::chrono::high_resolution_clock::now();
                // progress file
                trainPrg.correctPredPercent = static_cast<float>(100 * correctPredictions) / fileCount;
                trainPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
                trainPrg.timeTakenForTraining = previousTrainingTime + trainPrg.timeForCurrentSession;
                this->learningRate = trainPrg.currentLearningRate;
                trainPrg.sessionCount++;
                trainPrg.totalCycleCount += sessionFiles;
                trainPrg.trainingPredictions = correctPredictions;
                // confusion matrix
                confData = {};
                confData = confusionMatrixFunc(confusion);
                // stats
                computeStatsForCsv(cweights, bweights, cgradients, bgradients, activate, weightStats);
                // scores
                allScores.sse = allScores.totalSumOfError / trainPrg.filesProcessed;
                allScores.ssr = allScores.totalSumOfRegression / trainPrg.filesProcessed;
                allScores.sst = allScores.totalSumOfSquares / trainPrg.filesProcessed;
                allScores.r2 = allScores.ssr / allScores.sst;
                // save session data
                sessionDataToCsv(path2SessionDir, trainPrg.epoch, trainPrg.sessionCount, false,
                                    weightStats, confusion, confData, allScores, trainPrg);
                sessionEnd = 1;
                startTime = std::chrono::high_resolution_clock::now();
                std::cout << "Epoch: " << trainPrg.epoch << "\tFiles Processed: " << fileCount << "/" << totalFiles
                          << " \tPredictions: " << correctPredictions
                          << " \tTraining Accuracy: " << trainPrg.correctPredPercent << "%"
                          << " \tLoss: " << trainPrg.accLoss / static_cast<float>(trainPrg.filesProcessed)
                          << std::endl;
            }

            // If a session size is defined and reached, serialise weights
            if (sessionEnd == 1 || fileCount == totalFiles) {
                // computeStats(cweights, bweights, cgradients, bgradients, activate);
                if (logProgressToCSV(trainPrg, this->path2progress) != 1) {
                    throw std::runtime_error("Failed to log progress to CSV: " + this->path2progress);
                }
                serializeWeights(cweights, bweights, binFileAddress);
                filesInCurrentSession = 0;
                if (fileCount == totalFiles) {
                    std::cout << "All files processed. Ending training." << std::endl;
                    trainPrg.loss = trainPrg.accLoss / static_cast<float>(trainPrg.totalCycleCount);
                    bool notBatch = 0;
                    confData = {};
                    confData = confusionMatrixFunc(confusion);
                    allScores = {};
                    allScores.sse = allScores.totalSumOfError / totalFiles;
                    allScores.ssr = allScores.totalSumOfRegression / totalFiles;
                    allScores.sst = allScores.totalSumOfSquares / totalFiles;
                    allScores.r2 = allScores.ssr / allScores.sst;
                    epochDataToCsv(dataSetPath + "/mnn1d", trainPrg.epoch, notBatch, weightStats, confusion, confData, allScores,
                                    trainPrg, true);
                    break;
                }
            }
        }

        if(trainPrg.correctPredPercent >= 97.0f) {
            std::cout << "Training completed using minibatch of size " << BATCH_SIZE 
                      << "with accuracy of " << trainPrg.correctPredPercent << "%" << std::endl;
            break;
        }

        std::cout << "Training for next epoch: " << trainPrg.epoch + 1 << std::endl;
        trainPrg.epoch++;
        trainPrg.sessionSize = SESSION_SIZE;
        trainPrg.filesProcessed = 0;
        trainPrg.batchSize = 1;
        trainPrg.trainingPredictions = 0;
        trainPrg.currentLearningRate = this->learningRate;
        trainPrg.loss = 0;
        trainPrg.accLoss = 0;
        trainPrg.correctPredPercent = 0;
        trainPrg.sessionCount= 0;
        trainPrg.timeForCurrentSession = 0;
        logProgressToCSV(trainPrg, this->path2progress);

        allScores = {};
        confData = {};
        confusion.clear();
        confusion.assign(outWidth, std::vector<int>(outWidth, 0));
    }

    std::cout << "--- Training Finished (mnn2d) ---" << std::endl;
}