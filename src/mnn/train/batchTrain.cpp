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
 * @brief train network with mini batches over dataset epochs
 * @param dataSetPath path to dataset folder
 * @param useThreadOrBuffer use threads in CPU and full buffer-based operation in CUDA and OpenCL
 */
void mnn::miniBatchTraining(const std::string &dataSetPath, bool useThreadOrBuffer)
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
    std::cout << "\n--- Starting Mini-Batch Training (mnn) ---" << std::endl;
    std::cout << "Resuming from file index: " << this->trainPrg.filesProcessed << std::endl;

    // access training progress information from file address and use it to re-start training
    // from new session
    double previousTrainingTime = 0.0;
    batchSize = BATCH_SIZE;
    int curPreds = 0;
    if (!loadLastProgress(this->trainPrg, this->path2progress)) {
        // Preserve session and batch size set before calling train
        std::cout << "No progress file found or file is empty. Starting fresh training." << std::endl;
        this->trainPrg = {}; // Reset progress
        this->trainPrg.epoch = 0;
        this->trainPrg.sessionSize = SESSION_SIZE;
        this->trainPrg.batchSize = batchSize;
        this->trainPrg.currentLearningRate = this->learningRate;
        curPreds = 0;
    }
    else {
        std::cout << "Successfully loaded progress. Resuming training." << std::endl;
        this->learningRate = this->trainPrg.currentLearningRate;
        previousTrainingTime = this->trainPrg.timeTakenForTraining;
        curPreds = this->trainPrg.trainingPredictions;
        std::cout << "Found " << totalFiles << " files for training. Resuming from file index " << this->trainPrg.filesProcessed << "." << std::endl;
    }

    int fileCount = 0;
    int filesInCurrentSession = 0;
    // Update progress struct with file and batch info
    this->trainPrg.totalTrainFiles = totalFiles;
    this->trainPrg.batchSize = batchSize;
    int sessionFiles = this->trainPrg.sessionSize * this->trainPrg.batchSize;
    std::cout << "Batch Training with batch size: " << BATCH_SIZE << std::endl;
    std::cout << "Session Size (in batches/session): " << this->trainPrg.sessionSize << std::endl;
    std::cout << "Files in Single Session: " << sessionFiles << std::endl;

    // start time count
    auto startTime = std::chrono::high_resolution_clock::now();

    batchSize = BATCH_SIZE;
    this->learningRate = 0.02f;
    std::cout << "learning rate: " << this->learningRate << std::endl;
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
    unsigned int batchesProcessed = 0;
    int totalBatches = (totalFiles + batchSize - 1) / batchSize;
    std::cout << "Training with batch size: " << batchSize << " (" << totalBatches << " batches)" << std::endl;

    // Start iterating from the beginning of the batch where the last processed file was
    int startFileIndex = (this->trainPrg.filesProcessed / batchSize) * batchSize;
    fileCount = startFileIndex;
    std::cout << "Starting from file index " << startFileIndex << " to align with batches." << std::endl;

    while (1) {
        // Shuffle filePaths at the beginning of each epoch
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(filePaths.begin(), filePaths.end(), g);
        unsigned int correctPredictions = curPreds;
        for(int i = startFileIndex; i < totalFiles; i += batchSize) {
            std::vector<std::vector<float>> inBatch;
            std::vector<std::vector<float>> expBatch;
            int currentBatchEnd = std::min<int>(i + batchSize, totalFiles);
            // get image 
            for(int j = i; j < currentBatchEnd; ++j) {
                const auto& filePath = filePaths[j];
                inBatch.push_back(flatten(cvMat2vec(image2grey(filePath.string()))));
                std::string filename = filePath.stem().string();
                int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
                std::vector<float> exp(this->outSize, 0.0f);
                if (label < this->outSize) {
                    exp[label] = 1.0f;
                }
                expBatch.push_back(exp);
            }
            for(int j = 0; j < batchSize; j++) {
                for(int k = 0; k < inBatch[j].size(); k++) {
                    inBatch[j][k] /= 255.0f;
                }
            }

            inputBatch = inBatch;
            targetBatch = expBatch;
            // backend selection
            #ifdef USE_CPU
                trainBatch1c(inBatch, expBatch, useThreadOrBuffer);
            #elif USE_CU
                cuTrainBatch1c(inBatch, expBatch, useThreadOrBuffer);
            #elif USE_CL
                clTrainBatch1c(inBatch, expBatch, useThreadOrBuffer);
            #endif

            for (int j = 0; j < batchSize; j++) {
                if(maxIndex(outputBatch[j]) == maxIndex(expBatch[j])) {
                    correctPredictions++;
                }
                this->trainPrg.accLoss += crossEntropy(outputBatch[j], expBatch[j]);
            }

            // for progress tracking
            fileCount += batchSize;
            this->trainPrg.filesProcessed += batchSize;
            filesInCurrentSession += batchSize;
            bool sessionEnd = 0;
            if ((sessionFiles > 0 && filesInCurrentSession >= sessionFiles) || fileCount == totalFiles) {
                auto endTime = std::chrono::high_resolution_clock::now();
                this->trainPrg.correctPredPercent = static_cast<float>(100 * correctPredictions) / fileCount;
                this->trainPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
                this->trainPrg.trainingPredictions = correctPredictions;
                this->trainPrg.timeTakenForTraining += this->trainPrg.timeForCurrentSession;
                this->learningRate = this->trainPrg.currentLearningRate;
                this->trainPrg.totalSessionsOfTraining++;
                this->trainPrg.totalCycleCount += sessionFiles;
                this->trainPrg.loss = this->trainPrg.accLoss / static_cast<float>(this->trainPrg.filesProcessed);
                sessionEnd = 1;
                logProgressToCSV(this->trainPrg, this->path2progress);
                std::cout << "Epoch: " << this->trainPrg.epoch 
                          << " \tFiles: " << fileCount << "/" << totalFiles
                          << " \tPredictions: " << correctPredictions
                          << " \tTraining Accuracy: " << this->trainPrg.correctPredPercent << "%"
                          << " \tLoss: " << this->trainPrg.accLoss / static_cast<float>(this->trainPrg.filesProcessed)
                          << std::endl;
                serializeWeights(cweights, bweights, binFileAddress);
                filesInCurrentSession = 0;
                startTime = std::chrono::high_resolution_clock::now();
            }

            // If a session size is defined and reached, stop training for this session
            if (sessionEnd == 1 || fileCount == totalFiles) {
                // computeStats(cweights, bweights, cgradients, bgradients, activate);
                if (fileCount == totalFiles) {
                    std::cout << "All files processed. Next Epoch." << std::endl;
                    this->trainPrg.loss = this->trainPrg.accLoss / static_cast<float>(this->trainPrg.totalCycleCount);
                    break;
                }
            }
        }

        if(this->trainPrg.correctPredPercent >= 98.0f) {
            std::cout << "Training completed using minibatch of size " << BATCH_SIZE 
                      << "with accuracy of " << this->trainPrg.correctPredPercent << "%" << std::endl;
            break;
        }
        this->trainPrg.epoch += 1;
        this->trainPrg.filesProcessed = 0;
        this->trainPrg.correctPredPercent = 0;
        this->trainPrg.timeForCurrentSession = 0;
        this->trainPrg.trainingPredictions = 0;
        this->learningRate = this->trainPrg.currentLearningRate;
        this->trainPrg.loss = this->trainPrg.accLoss / static_cast<float>(this->trainPrg.filesProcessed);
        logProgressToCSV(this->trainPrg, this->path2progress);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    this->trainPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    this->trainPrg.timeTakenForTraining = previousTrainingTime + this->trainPrg.timeForCurrentSession;
    serializeWeights(cweights, bweights, binFileAddress);
    logProgressToCSV(this->trainPrg, this->path2progress);
    std::cout << "--- Training Finished (mnn) ---" << std::endl;
}


/**
 * @brief train network with mini batches over dataset epochs
 * @param dataSetPath path to dataset folder
 * @param useThreadOrBuffer use threads in CPU and full buffer-based operation in CUDA and OpenCL
 */
void mnn2d::miniBatchTraining(const std::string &dataSetPath, bool useThreadOrBuffer)
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
    std::cout << "\n--- Starting Mini-Batch Training (mnn) ---" << std::endl;
    std::cout << "Resuming from file index: " << this->trainPrg.filesProcessed << std::endl;

    // access training progress information from file address and use it to re-start training
    // from new session
    double previousTrainingTime = 0.0;
    batchSize = BATCH_SIZE;
    int curPreds = 0;
    if (!loadLastProgress(this->trainPrg, this->path2progress)) {
        // Preserve session and batch size set before calling train
        std::cout << "No progress file found or file is empty. Starting fresh training." << std::endl;
        this->trainPrg = {}; // Reset progress
        this->trainPrg.epoch = 0;
        this->trainPrg.sessionSize = SESSION_SIZE;
        this->trainPrg.batchSize = batchSize;
        this->trainPrg.currentLearningRate = this->learningRate;
        curPreds = 0;
    }
    else {
        std::cout << "Successfully loaded progress. Resuming training." << std::endl;
        this->learningRate = this->trainPrg.currentLearningRate;
        previousTrainingTime = this->trainPrg.timeTakenForTraining;
        curPreds = this->trainPrg.trainingPredictions;
        std::cout << "Found " << totalFiles << " files for training. Resuming from file index " << this->trainPrg.filesProcessed << "." << std::endl;
    }

    int fileCount = 0;
    int filesInCurrentSession = 0;
    // Update progress struct with file and batch info
    this->trainPrg.totalTrainFiles = totalFiles;
    this->trainPrg.batchSize = batchSize;
    int sessionFiles = this->trainPrg.sessionSize * this->trainPrg.batchSize;
    std::cout << "Batch Training with batch size: " << BATCH_SIZE << std::endl;
    std::cout << "Session Size (in batches/session): " << this->trainPrg.sessionSize << std::endl;
    std::cout << "Files in Single Session: " << sessionFiles << std::endl;

    // start time count
    auto startTime = std::chrono::high_resolution_clock::now();

    batchSize = BATCH_SIZE;
    this->learningRate = 0.005f;
    std::cout << "learning rate: " << this->learningRate << std::endl;
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
    unsigned int batchesProcessed = 0;
    int totalBatches = (totalFiles + batchSize - 1) / batchSize;
    std::cout << "Training with batch size: " << batchSize << " (" << totalBatches << " batches)" << std::endl;

    // Start iterating from the beginning of the batch where the last processed file was
    int startFileIndex = (this->trainPrg.filesProcessed / batchSize) * batchSize;
    fileCount = startFileIndex;
    std::cout << "Starting from file index " << startFileIndex << " to align with batches." << std::endl;

    while (1) {
        // Shuffle filePaths at the beginning of each epoch
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(filePaths.begin(), filePaths.end(), g);
        unsigned int correctPredictions = curPreds;
        for(int i = startFileIndex; i < totalFiles; i += batchSize) {
            std::vector<std::vector<std::vector<float>>> inBatch;
            std::vector<std::vector<float>> expBatch;
            int currentBatchEnd = std::min<int>(i + batchSize, totalFiles);
            // get image
            for(int j = i; j < currentBatchEnd; ++j) {
                const auto& filePath = filePaths[j];
                inBatch.push_back(cvMat2vec(image2grey(filePath.string())));
                std::string filename = filePath.stem().string();
                int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
                std::vector<float> exp(this->outWidth, 0.0f);
                if (label < this->outWidth) {
                    exp[label] = 1.0f;
                }
                expBatch.push_back(exp);
            }
            for(int j = 0; j < batchSize; j++) {
                for(int k = 0; k < inBatch[j].size(); k++) {
                    for(int p = 0; p < inBatch[j][k].size(); j++) {
                        inBatch[j][k][p] /= 255.0f;
                    }
                }
            }

            inputBatch = inBatch;
            targetBatch = expBatch;
            // backend selection
            #ifdef USE_CPU
                trainBatch1c(inBatch, expBatch, useThreadOrBuffer);
            #elif USE_CU
                cuTrainBatch1c(inBatch, expBatch, useThreadOrBuffer);
            #elif USE_CL
                clTrainBatch1c(inBatch, expBatch, useThreadOrBuffer);
            #endif

            for (int j = 0; j < batchSize; j++) {
                if(maxIndex(outputBatch[j]) == maxIndex(expBatch[j])) {
                    correctPredictions++;
                }
                this->trainPrg.accLoss += crossEntropy(outputBatch[j], expBatch[j]);
            }

            // for progress tracking
            fileCount += batchSize;
            this->trainPrg.filesProcessed += batchSize;
            filesInCurrentSession += batchSize;
            bool sessionEnd = 0;
            if (sessionFiles > 0 && filesInCurrentSession >= sessionFiles) {
                auto endTime = std::chrono::high_resolution_clock::now();
                this->trainPrg.correctPredPercent = static_cast<float>(100 * correctPredictions) / fileCount;
                this->trainPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
                this->trainPrg.timeTakenForTraining += this->trainPrg.timeForCurrentSession;
                this->learningRate = this->trainPrg.currentLearningRate;
                this->trainPrg.totalSessionsOfTraining++;
                this->trainPrg.totalCycleCount += sessionFiles;
                this->trainPrg.filesProcessed += sessionFiles;
                sessionEnd = 1;
                logProgressToCSV(this->trainPrg, this->path2progress);
                std::cout << "Epoch: " << this->trainPrg.epoch << "\tFiles: " << fileCount << "/" << totalFiles << " \tPredictions: " << correctPredictions << " \tTraining Accuracy: " << this->trainPrg.correctPredPercent << "%" 
                          << " \tLoss: " << this->trainPrg.accLoss / static_cast<float>(this->trainPrg.filesProcessed)<< std::endl;
            }

            // If a session size is defined and reached, stop training for this session
            if (sessionEnd == 1 || fileCount == totalFiles) {
                // computeStats(cweights, bweights, cgradients, bgradients, activate);
                serializeWeights(cweights, bweights, binFileAddress);
                filesInCurrentSession = 0;
                if (fileCount == totalFiles) {
                    std::cout << "All files processed. Ending training." << std::endl;
                    this->trainPrg.loss = this->trainPrg.accLoss / static_cast<float>(this->trainPrg.totalCycleCount);
                    break;
                }
            }
        }

        this->trainPrg.epoch += 1;
        if(this->trainPrg.correctPredPercent >= 97.0f) {
            std::cout << "Training completed using minibatch of size " << BATCH_SIZE 
                      << "with accuracy of " << this->trainPrg.correctPredPercent << "%" << std::endl;
            break;
        }
        this->trainPrg.correctPredPercent = 0;
        this->trainPrg.accLoss = 0;
        this->trainPrg.filesProcessed = 0;
        this->trainPrg.timeForCurrentSession = 0;
        this->trainPrg.loss = 0;
        this->trainPrg.timeForCurrentSession = 0;
        logProgressToCSV(this->trainPrg, this->path2progress);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    this->trainPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    this->trainPrg.timeTakenForTraining = previousTrainingTime + this->trainPrg.timeForCurrentSession;
    serializeWeights(cweights, bweights, binFileAddress);
    logProgressToCSV(this->trainPrg, this->path2progress);
    std::cout << "--- Training Finished (mnn) ---" << std::endl;
}