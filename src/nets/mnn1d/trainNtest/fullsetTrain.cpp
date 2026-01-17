#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include <random>
#include "mnn1d.hpp"

// for MNN

/**
 * @brief single cycle multi epoch online training
 * @param dataSetPath path to dataset folder
 * @param useThreadOrBuffer use threads in CPU and full buffer-based operation in CUDA and OpenCL
 * @param isRGB if image is RGB 1, else 0 for grey image
 */
void mnn1d::fullDataSetTraining(const std::string &dataSetPath, bool isRGB, bool useThreadOrBuffer)
{
    std::cout << "This is fullDataSetTraining.\n";
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
    if (weightUpdateType == 0) std::cout << "> Weight Update Type: SGD" << std::endl;
    else if (weightUpdateType == 1) std::cout << "> Weight Update Type: L1 (lasso) reg." << std::endl;
    else if (weightUpdateType == 2) std::cout << "> Weight Update Type: L2 (ridge) reg." << std::endl;
    else if (weightUpdateType == 3) std::cout << "> Weight Update Type: L1+L2 (elastic net) reg." << std::endl;
    else if (weightUpdateType == 4) std::cout << "> Weight Update Type: Weight Decay." << std::endl;
    else if (weightUpdateType == 5) std::cout << "> Weight Update Type: Dropout" << std::endl;

    // Sort the dataset to ensure consistent order for resumable training
    std::sort(filePaths.begin(), filePaths.end());
    int totalFiles = filePaths.size();
    std::cout << "\n--- Starting Full Dataset Training (mnn1d) ---" << std::endl;

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
    int epochCount = 1;
    int curPreds = 0;
    int fileCount = 0;
    allScores = {};
    confusion.assign(outSize, std::vector<int>(outSize, 0));
    if (!loadLastProgress(trainPrg, this->path2progress)) {
        std::cout << "No progress file found or file is empty. Starting fresh training." << std::endl;
        trainPrg = {}; // Reset progress
        confData = {};
        trainPrg.sessionSize = ((totalFiles % SESSION_SIZE == 0) && (totalFiles / SESSION_SIZE <= 100)) ? SESSION_SIZE : (totalFiles / 100);
        trainPrg.batchSize = 1;
        trainPrg.currentLearningRate = this->learningRate;
        curPreds = 0;
        fileCount = 0;
        trainPrg.filesProcessed = 0;
    }
    else {
        std::cout << "Successfully loaded progress." << std::endl;
        if(trainPrg.filesProcessed == 0)
            std::cout << "Resuming Training!" << std::endl;
        else {
            std::cout << "Starting from file index " << trainPrg.filesProcessed << " with epoch " << trainPrg.epoch << std::endl;
            std::string lastSessionFile = path2SessionDir + "/e" + std::to_string(trainPrg.epoch) + "_s_" + std::to_string(trainPrg.sessionCount) + "_b_0.csv";
            if (trainPrg.sessionCount != 0) {
                if (loadSessionConfusionMatrix(confusion, lastSessionFile))
                    std::cout << "Loaded confusion matrix from " << lastSessionFile << std::endl;
                else 
                    std::cout << "Warning: Could not load confusion matrix from " << lastSessionFile << std::endl;
            }
        }

        this->learningRate = trainPrg.currentLearningRate;
        previousTrainingTime = trainPrg.timeTakenForTraining;
        std::cout << "Found " << totalFiles << " files for training. Resuming from file index " << trainPrg.filesProcessed << "." << std::endl;
        curPreds = trainPrg.trainingPredictions;
        epochs = trainPrg.epoch;
        epochCount = trainPrg.epoch;
    }

    int filesInCurrentSession = 0;
    // Update progress struct with file and batch info
    int sessionFiles = trainPrg.sessionSize; // * trainPrg.batchSize;
    std::cout << "Session Size (in batches/session): " << trainPrg.sessionSize << std::endl;
    std::cout << "Files in Single Session: " << sessionFiles << std::endl;
    std::cout << "learning rate: " << learningRate << std::endl;

    // Resize confusion matrix before use
    auto startTime = std::chrono::high_resolution_clock::now();
    unsigned int correctPredictions = curPreds;

    while (1) {
        // Shuffle filePaths at the beginning of each epoch
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(filePaths.begin(), filePaths.end(), g);
        for(const auto& filePath : filePaths) {
            // Skip files that have already been processed in previous sessions
            if (fileCount < trainPrg.filesProcessed) {
                fileCount++;
                continue;
            }

            // Convert image to a flat 1D vector
            std::vector<float> in = flatten(image2matrix(filePath.string(), isRGB));
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
                train1c(in, target, useThreadOrBuffer);
            #elif USE_CU
                cuTrain1c(in, target, useThreadOrBuffer);
            #elif USE_CL
                clTrain1c(in, target, useThreadOrBuffer);
            #endif

            fileCount++;
            trainPrg.filesProcessed++;
            filesInCurrentSession++;
            if (label < confusion.size() && maxIndex(output) < confusion[0].size()) {
                confusion[label][maxIndex(output)]++;
            }
            if(maxIndex(output) == maxIndex(target))
                correctPredictions++;
            trainPrg.accLoss += crossEntropy(output, target);
            getScore(output, target, allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);

            bool sessionEnd = 0;
            if ((sessionFiles > 0 && filesInCurrentSession == sessionFiles) || fileCount == totalFiles) {
                auto endTime = std::chrono::high_resolution_clock::now();
                // progress file
                trainPrg.correctPredPercent = static_cast<float>(100 * correctPredictions) / fileCount;
                trainPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
                trainPrg.timeTakenForTraining = previousTrainingTime + trainPrg.timeForCurrentSession;
                trainPrg.currentLearningRate = this->learningRate;
                trainPrg.sessionCount++;
                trainPrg.totalCycleCount += filesInCurrentSession;
                trainPrg.trainingPredictions = correctPredictions;
                // confusion matrix
                confData = {};
                confData = confusionMatrixFunc(confusion);
                // scores
                allScores.sse = allScores.totalSumOfError / trainPrg.filesProcessed;
                allScores.ssr = allScores.totalSumOfRegression / trainPrg.filesProcessed;
                allScores.sst = allScores.totalSumOfSquares / trainPrg.filesProcessed;
                allScores.r2 = allScores.ssr / allScores.sst;
                // stats
                computeStatsForCsv(cweights, bweights, cgradients, bgradients, activate, weightStats);
                sessionDataToCsv(path2SessionDir, trainPrg.epoch, trainPrg.sessionCount, false,
                                    weightStats, confusion, confData, allScores, trainPrg);
                sessionEnd = 1;
                startTime = std::chrono::high_resolution_clock::now();
                std::cout << "Epoch: " << trainPrg.epoch << "\tFiles Processed: " << fileCount << "/" << totalFiles
                          << "\tPredictions: " << correctPredictions
                          << "\tTraining Accuracy: " << trainPrg.correctPredPercent << "%"
                          << "\tLoss: " << trainPrg.accLoss / static_cast<float>(trainPrg.filesProcessed)
                          << std::endl;
            }

            // If a session size is defined and reached, serialise weights
            if (sessionEnd == 1 || fileCount == totalFiles) {
                // computeStats(cweights, bweights, cgradients, bgradients, activate);
                if (fileCount == totalFiles) trainPrg.loss = trainPrg.accLoss / static_cast<float>(totalFiles);
                if (logProgressToCSV(trainPrg, this->path2progress) != 1) {
                    throw std::runtime_error("Failed to log progress to CSV: " + this->path2progress);
                }
                serializeWeights(cweights, bweights, binFileAddress);
                filesInCurrentSession = 0;
                if (fileCount == totalFiles) {
                    std::cout << "All files processed. Ending training for this epoch." << std::endl;
                    bool notBatch = 0;
                    confData = {};
                    confData = confusionMatrixFunc(confusion);
                    allScores.sse = allScores.totalSumOfError / totalFiles;
                    allScores.ssr = allScores.totalSumOfRegression / totalFiles;
                    allScores.sst = allScores.totalSumOfSquares / totalFiles;
                    allScores.r2 = allScores.ssr / allScores.sst;
                    weightStats.clear();
                    computeStatsForCsv(cweights, bweights, weightStats);
                    epochDataToCsv1(dataSetPath + "/mnn1d/epoch", trainPrg.epoch, notBatch, weightStats, confusion, confData, allScores,
                                   trainPrg, 1);
                    std::string epochWeight = dataSetPath + "/mnn1d/epochWeights" + std::to_string(trainPrg.epoch) + ".bin";
                    createBinFile(epochWeight, param);
                    serializeWeights(cweights, bweights, epochWeight);
                    break;
                }
            }
        }

        fileCount = 0;
        trainPrg.filesProcessed = 0;
        if(trainPrg.correctPredPercent >= 99.0f) {
            std::cout << "Training completed using online training with accuracy of " << trainPrg.correctPredPercent << "% after "
                      << trainPrg.epoch << " epochs." << std::endl;
            break;
        }

        if (epochCount == EPOCH) {
            std::cout << "Training Completed at accuracy " << trainPrg.correctPredPercent << "% at epoch " << trainPrg.epoch << "." << std::endl;
            break;
        }
        if ((epochCount >= 4 && epochCount < EPOCH) || trainPrg.correctPredPercent >= 90.0f) {
            learningRate = cosineAnnealing(LEARNING_MAX, LEARNING_MIN, epochCount, EPOCH);
            std::cout << "New Learning Rate For Epoch " << epochCount << " : " << learningRate << std::endl;
        }
        epochCount++;

        std::cout << "Training for next epoch: " << trainPrg.epoch + 1 << std::endl;
        trainPrg.epoch++;
        trainPrg.sessionSize = ((totalFiles % SESSION_SIZE == 0) && (totalFiles / SESSION_SIZE <= 100)) ? SESSION_SIZE : (totalFiles / 100);
        trainPrg.filesProcessed = 0;
        trainPrg.batchSize = 1;
        trainPrg.trainingPredictions = 0;
        trainPrg.currentLearningRate = this->learningRate;
        trainPrg.loss = 0;
        trainPrg.accLoss = 0;
        trainPrg.correctPredPercent = 0;
        trainPrg.sessionCount = 0;
        trainPrg.timeForCurrentSession = 0;
        logProgressToCSV(trainPrg, this->path2progress);
        correctPredictions = 0;
        allScores = {};
        confData = {};
        confusion.clear();
        confusion.assign(outSize, std::vector<int>(outSize, 0));
    }

    std::cout << "--- Training Finished (mnn1d) ---" << std::endl;
}