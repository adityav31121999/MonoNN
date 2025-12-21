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

/**
 * IMP: Online training with batches fails drastically and is not good.
 * Hence, removed those parts from the online training definition.
 */

// for MNN

/**
 * @brief train network online on given dataset
 * @param dataSetPath path to dataset folder
 * @param useThreadOrBuffer use threads in CPU and full buffer-based operation in CUDA and OpenCL
 */
void mnn::onlineTraining(const std::string &dataSetPath, bool useThreadOrBuffer)
{
    // Access all image files from the dataset path
    std::vector<std::filesystem::path> filePaths;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(dataSetPath)) {
            if (entry.is_regular_file()) {
                filePaths.push_back(entry.path());
            }
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Failed to read dataset directory: " + std::string(e.what()));
    }

    if (filePaths.empty()) {
        std::cout << "Warning: No files found in dataset directory: " << dataSetPath << std::endl;
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

    // Sort the dataset to ensure consistent order for resumable training
    std::sort(filePaths.begin(), filePaths.end());

    int totalFiles = filePaths.size();
    std::cout << "\n--- Starting Training (mnn) ---" << std::endl;
    std::cout << "Resuming from file index: " << trainPrg.filesProcessed << std::endl;

    // access training progress information from file address and use it to re-start training
    // from new session
    double previousTrainingTime = 0.0;
    if (!loadLastProgress(trainPrg, this->path2progress)) {
        // Preserve session and batch size set before calling train
        std::cout << "No progress file found or file is empty. Starting fresh training." << std::endl;
        trainPrg = {}; // Reset progress
        trainPrg.sessionSize = ((totalFiles % SESSION_SIZE == 0) && (totalFiles % SESSION_SIZE <= 100)) ? SESSION_SIZE : totalFiles % 100;
        trainPrg.batchSize = 1;
        trainPrg.currentLearningRate = this->learningRate;
    }
    else {
        std::cout << "Successfully loaded progress. Resuming training." << std::endl;
        this->learningRate = trainPrg.currentLearningRate;
        previousTrainingTime = trainPrg.timeTakenForTraining;
        std::cout << "Found " << totalFiles << " files for training. Resuming from file index " << trainPrg.filesProcessed << "." << std::endl;
    }

    int fileCount = 0;
    int filesInCurrentSession = 0;
    // Update progress struct with file and batch info
    trainPrg.totalTrainFiles = totalFiles;
    trainPrg.batchSize = batchSize;
    int sessionFiles = trainPrg.sessionSize * trainPrg.batchSize;
    std::cout << "Session Size (in batches/session): " << trainPrg.sessionSize << std::endl;
    std::cout << "Files in Single Session: " << sessionFiles << std::endl;

    // start time count
    auto startTime = std::chrono::high_resolution_clock::now();

    batchSize = 1;
    learningRate = 0.001f;
    std::cout << "learning rate: " << learningRate << std::endl;
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
        for(int i = 0; i < in.size(); i++) {
            in[i] /= 255;
        }
        this->input = in;
        if (maxIndex(output) == maxIndex(exp)) trainPrg.trainingPredictions++;
        if (label < confusion.size() && maxIndex(output) < confusion[0].size()) {
            confusion[label][maxIndex(output)]++;
        }
        trainPrg.accLoss += crossEntropy(output, exp);
        getScore(output, exp, allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);
        this->target = exp;
        // backend selection
        #ifdef USE_CPU
            (useThreadOrBuffer == 0) ? train(in, exp) : threadTrain(in, exp);
        #elif USE_CU
            (useThreadOrBuffer == 0) ? cuTrain(in, exp) : cuBufTrain(in, exp);
        #elif USE_CL
            (useThreadOrBuffer == 0) ? clTrain(in, exp) : clBufTrain(in, exp);
        #endif

        // for progress tracking
        fileCount++;
        trainPrg.filesProcessed++;
        filesInCurrentSession++;
        bool sessionEnd = 0;
        if ((sessionFiles > 0 && filesInCurrentSession == trainPrg.sessionSize) || fileCount == totalFiles) {
            std::cout << "Session file limit (" << trainPrg.sessionSize << ") reached or End of Epoch." << std::endl;
            auto endTime = std::chrono::high_resolution_clock::now();
            trainPrg.correctPredPercent = static_cast<float>(100 * trainPrg.trainingPredictions) / fileCount;
            trainPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
            trainPrg.timeTakenForTraining = previousTrainingTime + trainPrg.timeForCurrentSession;
            this->learningRate = trainPrg.currentLearningRate;
            trainPrg.sessionCount++;
            trainPrg.totalCycleCount += filesInCurrentSession;
            // Confusion Matrix
            confData = {};
            confData = confusionMatrixFunc(confusion);
            // Stats
            computeStatsForCsv(cweights, bweights, cgradients, bgradients, activate, weightStats);
            // Scores
            allScores.sse = allScores.totalSumOfError / trainPrg.filesProcessed;
            allScores.ssr = allScores.totalSumOfRegression / trainPrg.filesProcessed;
            allScores.sst = allScores.totalSumOfSquares / trainPrg.filesProcessed;
            allScores.r2 = allScores.ssr / allScores.sst;
            // Save session data
            sessionDataToCsv(path2SessionDir, trainPrg.epoch, trainPrg.sessionCount, false,
                                weightStats, confusion, confData, allScores, trainPrg);
            trainPrg.loss = trainPrg.accLoss / static_cast<float>(trainPrg.filesProcessed);
            sessionEnd = 1;
            startTime = std::chrono::high_resolution_clock::now();
        }
        std::cout<< "File count: " << fileCount << std::endl;

        // If a session size is defined and reached, stop training for this session
        if (sessionEnd == 1 || fileCount == totalFiles) {
            std::cout << "Processed " << fileCount << "/" << totalFiles << " files..." << std::endl;
            std::cout << "====== Diagnostics For Current Session ======" << std::endl;
            if (logProgressToCSV(trainPrg, this->path2progress) == 1)
                std::cout << "Progress logged successfully." << std::endl;
            else 
                throw std::runtime_error("Failed to log progress to CSV: " + this->path2progress);
            serializeWeights(cweights, bweights, binFileAddress);
            std::cout << "============== To Next Session ==============" << std::endl;
            filesInCurrentSession = 0; // Reset for the next session
            if (fileCount == totalFiles) {
                std::cout << "All files processed. Ending training." << std::endl;
                trainPrg.loss = trainPrg.accLoss / static_cast<float>(fileCount);
                bool notBatch = 0;
                confData = {};
                confData = confusionMatrixFunc(confusion);
                allScores.sse = allScores.totalSumOfError / trainPrg.totalCycleCount;
                allScores.ssr = allScores.totalSumOfRegression / trainPrg.totalCycleCount;
                allScores.sst = allScores.totalSumOfSquares / trainPrg.totalCycleCount;
                allScores.r2 = allScores.ssr / allScores.sst;
                epochDataToCsv(dataSetPath + "/mnn1d", trainPrg.epoch, notBatch,
                                weightStats, confusion, confData, allScores, trainPrg,
                                true);
                break;
            }
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    trainPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    trainPrg.timeTakenForTraining = previousTrainingTime + trainPrg.timeForCurrentSession;
    serializeWeights(cweights, bweights, binFileAddress);
    logProgressToCSV(trainPrg, this->path2progress);
    std::cout << "--- Training Finished (mnn) ---" << std::endl;
}

// for MNN2D

/**
 * @brief train network online on given dataset
 * @param dataSetPath path to dataset folder.
 * @param useThreadOrBuffer use threads in CPU and full buffer-based operation in CUDA and OpenCL
 */
void mnn2d::onlineTraining(const std::string &dataSetPath, bool useThreadOrBuffer)
{
    // Access all image files from the dataset path
    std::vector<std::filesystem::path> filePaths;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(dataSetPath)) {
            if (entry.is_regular_file()) {
                filePaths.push_back(entry.path());
            }
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Failed to read dataset directory: " + std::string(e.what()));
    }

    if (filePaths.empty()) {
        std::cout << "Warning: No files found in dataset directory: " << dataSetPath << std::endl;
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

    // Sort the dataset to ensure consistent order for resumable training
    std::sort(filePaths.begin(), filePaths.end());

    int totalFiles = filePaths.size();
    std::cout << "\n--- Starting Training (mnn) ---" << std::endl;
    std::cout << "Resuming from file index: " << trainPrg.filesProcessed << std::endl;

    // access training progress information from file address and use it to re-start training
    // from new session
    double previousTrainingTime = 0.0;
    if (!loadLastProgress(trainPrg, this->path2progress)) {
        // Preserve session and batch size set before calling train
        std::cout << "No progress file found or file is empty. Starting fresh training." << std::endl;
        trainPrg = {}; // Reset progress
        trainPrg.sessionSize = ((totalFiles % SESSION_SIZE == 0) && (totalFiles % SESSION_SIZE <= 100)) ? SESSION_SIZE : totalFiles % 100;
        trainPrg.batchSize = 1;
        trainPrg.currentLearningRate = this->learningRate;
    }
    else {
        std::cout << "Successfully loaded progress. Resuming training." << std::endl;
        this->learningRate = trainPrg.currentLearningRate; // Use the learning rate from the last session
        previousTrainingTime = trainPrg.timeTakenForTraining; // Carry over total time
    }
    std::cout << "Found " << totalFiles << " files for training. Resuming from file index " << trainPrg.filesProcessed << "." << std::endl;

    int fileCount = 0;
    int filesInCurrentSession = 0;
    // Update progress struct with file and batch info
    trainPrg.totalTrainFiles = totalFiles;
    trainPrg.batchSize = batchSize;
    int sessionFiles = trainPrg.sessionSize * trainPrg.batchSize;
    std::cout << "Session Size (in batches/session): " << trainPrg.sessionSize << std::endl;
    std::cout << "Files in Single Session: " << sessionFiles << std::endl;

    // start time count
    auto startTime = std::chrono::high_resolution_clock::now();

    batchSize = 1;
    learningRate = 0.001f;
    std::cout << "learning rate: " << learningRate << std::endl;
    for(const auto& filePath : filePaths) {
        // Skip files that have already been processed in previous sessions
        if (fileCount < trainPrg.filesProcessed) {
            fileCount++;
            continue;
        }

        // Convert image to a 2D vector
        std::vector<std::vector<float>> in = cvMat2vec(image2grey(filePath.string()));

        // Extract label and create one-hot target vector
        std::string filename = filePath.stem().string();
        int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
        std::vector<float> exp(this->outWidth, 0.0f);
        if (label < this->outWidth) {
            exp[label] = 1.0f;
        }
        for(int i = 0; i < in.size(); i++) {
            for(int j = 0; j < in[i].size(); j++) {
                in[i][j] /= 255;
            }
        }

        if (maxIndex(output) == maxIndex(exp)) trainPrg.trainingPredictions++;
        if (label < confusion.size() && maxIndex(output) < confusion[0].size()) {
            confusion[label][maxIndex(output)]++;
        }
        trainPrg.accLoss += crossEntropy(output, exp);
        getScore(output, exp, allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);
        target = exp;
        #ifdef USE_CPU
            (useThreadOrBuffer == 0) ? train(in, exp) : threadTrain(in, exp);
        #elif USE_CU
            (useThreadOrBuffer == 0) ? cuTrain(in, exp) : cuBufTrain(in, exp);
        #elif USE_CL
            (useThreadOrBuffer == 0) ? clTrain(in, exp) : clBufTrain(in, exp);
        #endif

        // for progress tracking
        fileCount++;
        trainPrg.filesProcessed++;
        filesInCurrentSession++;
        bool sessionEnd = 0;
        if ((sessionFiles > 0 && filesInCurrentSession == trainPrg.sessionSize) || fileCount == totalFiles) {
            std::cout << "Session file limit (" << trainPrg.sessionSize << ") reached or End of Epoch." << std::endl;
            auto endTime = std::chrono::high_resolution_clock::now();
            trainPrg.correctPredPercent = static_cast<float>(100 * trainPrg.trainingPredictions) / fileCount;
            trainPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
            trainPrg.timeTakenForTraining = previousTrainingTime + trainPrg.timeForCurrentSession;
            this->learningRate = trainPrg.currentLearningRate;
            trainPrg.sessionCount++;
            trainPrg.totalCycleCount += filesInCurrentSession;
            // Confusion Matrix
            confData = {};
            confData = confusionMatrixFunc(confusion);
            // Stats
            computeStatsForCsv(cweights, bweights, cgradients, bgradients, activate, weightStats);
            // Scores
            allScores.sse = allScores.totalSumOfError / trainPrg.filesProcessed;
            allScores.ssr = allScores.totalSumOfRegression / trainPrg.filesProcessed;
            allScores.sst = allScores.totalSumOfSquares / trainPrg.filesProcessed;
            allScores.r2 = allScores.ssr / allScores.sst;
            // Save session data
            sessionDataToCsv(path2SessionDir, trainPrg.epoch, trainPrg.sessionCount, false,
                                weightStats, confusion, confData, allScores, trainPrg);
            trainPrg.loss = trainPrg.accLoss / static_cast<float>(trainPrg.filesProcessed);
            sessionEnd = 1;
            startTime = std::chrono::high_resolution_clock::now();
        }
        std::cout<< "File count: " << fileCount << std::endl;

        // If a session size is defined and reached, stop training for this session
        if (sessionEnd == 1 || fileCount == totalFiles) {
            std::cout << "Processed " << fileCount << "/" << totalFiles << " files..." << std::endl;
            std::cout << "====== Diagnostics For Current Session ======" << std::endl;
            if (logProgressToCSV(trainPrg, this->path2progress) == 1)
                std::cout << "Progress logged successfully." << std::endl;
            else 
                throw std::runtime_error("Failed to log progress to CSV: " + this->path2progress);
            serializeWeights(cweights, bweights, binFileAddress);
            std::cout << "============== To Next Session ==============" << std::endl;
            filesInCurrentSession = 0; // Reset for the next session
            if (fileCount == totalFiles) {
                std::cout << "All files processed. Ending training." << std::endl;
                trainPrg.loss = trainPrg.accLoss / static_cast<float>(fileCount);
                bool notBatch = 0;
                confData = {};
                confData = confusionMatrixFunc(confusion);
                allScores.sse = allScores.totalSumOfError / trainPrg.totalCycleCount;
                allScores.ssr = allScores.totalSumOfRegression / trainPrg.totalCycleCount;
                allScores.sst = allScores.totalSumOfSquares / trainPrg.totalCycleCount;
                allScores.r2 = allScores.ssr / allScores.sst;
                epochDataToCsv(dataSetPath + "/mnn2d", trainPrg.epoch, notBatch,
                                weightStats, confusion, confData, allScores, trainPrg,
                                true);
                break;
            }
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    trainPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    trainPrg.timeTakenForTraining = previousTrainingTime + trainPrg.timeForCurrentSession;
    serializeWeights(cweights, bweights, binFileAddress);
    logProgressToCSV(trainPrg, this->path2progress);
    std::cout << "--- Training Finished (mnn2d) ---" << std::endl;
}