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
 * @brief train network on given dataset
 * @param dataSetPath path to dataset folder
 * @param isBatchTrain whether to use batch training or not
 */
void mnn::train(const std::string &dataSetPath, bool isBatchTrain)
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
    std::cout << "Resuming from file index: " << this->mnnPrg.filesProcessed << std::endl;

    // access training progress information from file address and use it to re-start training
    // from new session
    double previousTrainingTime = 0.0;
    if (!loadLastProgress(this->mnnPrg, this->path2progress)) {
        // Preserve session and batch size set before calling train
        unsigned int sessionSizeBackup = this->mnnPrg.sessionSize;
        unsigned int batchSizeBackup = this->batchSize;
        std::cout << "No progress file found or file is empty. Starting fresh training." << std::endl;
        this->mnnPrg = {}; // Reset progress
        this->mnnPrg.sessionSize = SESSION_SIZE;
        this->mnnPrg.batchSize = batchSize;
        this->mnnPrg.currentLearningRate = this->learningRate;
        this->mnnPrg.sessionSize = sessionSizeBackup;
        this->mnnPrg.batchSize = batchSizeBackup;
    }
    else {
        std::cout << "Successfully loaded progress. Resuming training." << std::endl;
        this->learningRate = this->mnnPrg.currentLearningRate; // Use the learning rate from the last session
        previousTrainingTime = this->mnnPrg.timeTakenForTraining; // Carry over total time
    }
    std::cout << "Found " << totalFiles << " files for training. Resuming from file index " << this->mnnPrg.filesProcessed << "." << std::endl;

    int fileCount = 0;
    int filesInCurrentSession = 0;
    // Update progress struct with file and batch info
    this->mnnPrg.totalTrainFiles = totalFiles;
    this->mnnPrg.batchSize = batchSize;
    int sessionFiles = this->mnnPrg.sessionSize * this->mnnPrg.batchSize;
    std::cout << "Training with batch size: " << BATCH_SIZE << std::endl;
    std::cout << "Session Size (in batches/session): " << this->mnnPrg.sessionSize << std::endl;
    std::cout << "Files in Single Session: " << sessionFiles << std::endl;

    // start time count
    auto startTime = std::chrono::high_resolution_clock::now();

    // Train based on batch size
    if (isBatchTrain == false) {
        batchSize = 1;
        learningRate = 0.001f;
        std::cout << "learning rate: " << learningRate << std::endl;
        for(const auto& filePath : filePaths) {
            // Skip files that have already been processed in previous sessions
            if (fileCount < this->mnnPrg.filesProcessed) {
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
            input = in, target = exp;
            // backend selection
            #ifdef USE_CPU
                train(in, exp);
            #elif USE_CU
                cuTrain(in, exp);
            #elif USE_CL
                clTrain(in, exp);
            #endif

            // for progress tracking
            fileCount++;
            this->mnnPrg.filesProcessed++;
            filesInCurrentSession++;
            bool sessionEnd = 0;
            if (sessionFiles > 0 && filesInCurrentSession == this->mnnPrg.sessionSize) {
                std::cout << "Session file limit (" << this->mnnPrg.sessionSize << ") reached." << std::endl;
                auto endTime = std::chrono::high_resolution_clock::now();
                this->mnnPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
                this->mnnPrg.timeTakenForTraining = previousTrainingTime + this->mnnPrg.timeForCurrentSession;
                this->learningRate = this->mnnPrg.currentLearningRate;
                this->mnnPrg.totalSessionsOfTraining++;
                sessionEnd = 1;
            }
            std::cout<< "File count: " << fileCount << std::endl;

            // If a session size is defined and reached, stop training for this session
            if (sessionEnd == 1 || fileCount == totalFiles) {
                std::cout << "Processed " << fileCount << "/" << totalFiles << " files..." << std::endl;
                std::cout << "====== Diagnostics For Current Session ======" << std::endl;
                computeStats(cweights, bweights, cgradients, bgradients, activate);
                if (logProgressToCSV(this->mnnPrg, this->path2progress) == 1)
                    std::cout << "Progress logged successfully." << std::endl;
                else 
                    throw std::runtime_error("Failed to log progress to CSV: " + this->path2progress);
                serializeWeights(cweights, bweights, binFileAddress);
                std::cout << "============== To Next Session ==============" << std::endl;
                filesInCurrentSession = 0; // Reset for the next session
                if (fileCount == totalFiles) {
                    std::cout << "All files processed. Ending training." << std::endl;
                    this->mnnPrg.loss = this->mnnPrg.accLoss / static_cast<float>(this->mnnPrg.totalCycleCount);
                    break;
                }
            }
        }
    }
    else {
        batchSize = BATCH_SIZE;
        this->learningRate = 0.01f;
        std::cout << "learning rate for batch size: " << this->learningRate << std::endl;
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
        int startFileIndex = (this->mnnPrg.filesProcessed / batchSize) * batchSize;
        std::cout << "Starting from file index " << startFileIndex << " to align with batches." << std::endl;

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
            // backend selection
            #ifdef USE_CPU
                trainBatch(inBatch, expBatch);
            #elif USE_CU
                cuTrainBatch(inBatch, expBatch);
            #elif USE_CL
                clTrainBatch(inBatch, expBatch);
            #endif

            // for progress tracking
            fileCount += batchSize;
            this->mnnPrg.filesProcessed += batchSize;
            filesInCurrentSession += batchSize;
            bool sessionEnd = 0;
            if (sessionFiles > 0 && filesInCurrentSession == this->mnnPrg.sessionSize) {
                std::cout << "Session file limit (" << this->mnnPrg.sessionSize << ") reached." << std::endl;
                auto endTime = std::chrono::high_resolution_clock::now();
                this->mnnPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
                this->mnnPrg.timeTakenForTraining = previousTrainingTime + this->mnnPrg.timeForCurrentSession;
                this->mnnPrg.currentLearningRate = learningRate;
                this->mnnPrg.totalSessionsOfTraining++;
                sessionEnd = 1;
            }
            std::cout<< "File count: " << fileCount << std::endl;

            // If a session size is defined and reached, stop training for this session
            if (sessionEnd == 1 || fileCount == totalFiles) {
                std::cout << "Processed " << fileCount << "/" << totalFiles << " files..." << std::endl;
                std::cout << "====== Diagnostics For Current Session ======" << std::endl;
                if (logProgressToCSV(this->mnnPrg, this->path2progress) == 1)
                    std::cout << "Progress logged successfully." << std::endl;
                else 
                    throw std::runtime_error("Failed to log progress to CSV: " + this->path2progress);
                serializeWeights(cweights, bweights, binFileAddress);
                std::cout << "============== To Next Session ==============" << std::endl;
                filesInCurrentSession = 0; // Reset for the next session
                if (fileCount == totalFiles) {
                    std::cout << "All files processed. Ending training." << std::endl;
                    this->mnnPrg.loss = this->mnnPrg.accLoss / static_cast<float>(this->mnnPrg.totalCycleCount);
                    break;
                }
            }
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    this->mnnPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    this->mnnPrg.timeTakenForTraining = previousTrainingTime + this->mnnPrg.timeForCurrentSession;
    serializeWeights(cweights, bweights, binFileAddress);
    logProgressToCSV(this->mnnPrg, this->path2progress);
    std::cout << "--- Training Finished (mnn) ---" << std::endl;
}


/**
 * @brief train network on given dataset
 * @param dataSetPath path to dataset folder.
 * @param isBatchTrain whether to use batch training or not
 */
void mnn2d::train(const std::string &dataSetPath, bool isBatchTrain)
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
    std::cout << "Resuming from file index: " << this->mnn2dPrg.filesProcessed << std::endl;

    // access training progress information from file address and use it to re-start training
    // from new session
    double previousTrainingTime = 0.0;
    if (!loadLastProgress(this->mnn2dPrg, this->path2progress)) {
        // Preserve session and batch size set before calling train
        unsigned int sessionSizeBackup = this->mnn2dPrg.sessionSize;
        unsigned int batchSizeBackup = this->batchSize;
        std::cout << "No progress file found or file is empty. Starting fresh training." << std::endl;
        this->mnn2dPrg = {}; // Reset progress
        this->mnn2dPrg.sessionSize = SESSION_SIZE;
        this->mnn2dPrg.batchSize = batchSize;
        this->mnn2dPrg.currentLearningRate = this->learningRate;
        this->mnn2dPrg.sessionSize = sessionSizeBackup;
        this->mnn2dPrg.batchSize = batchSizeBackup;
    }
    else {
        std::cout << "Successfully loaded progress. Resuming training." << std::endl;
        this->learningRate = this->mnn2dPrg.currentLearningRate; // Use the learning rate from the last session
        previousTrainingTime = this->mnn2dPrg.timeTakenForTraining; // Carry over total time
    }
    std::cout << "Found " << totalFiles << " files for training. Resuming from file index " << this->mnn2dPrg.filesProcessed << "." << std::endl;

    int fileCount = 0;
    int filesInCurrentSession = 0;
    // Update progress struct with file and batch info
    this->mnn2dPrg.totalTrainFiles = totalFiles;
    this->mnn2dPrg.batchSize = batchSize;
    int sessionFiles = this->mnn2dPrg.sessionSize * this->mnn2dPrg.batchSize;
    std::cout << "Training with batch size: " << BATCH_SIZE << std::endl;
    std::cout << "Session Size (in batches/session): " << this->mnn2dPrg.sessionSize << std::endl;
    std::cout << "Files in Single Session: " << sessionFiles << std::endl;

    // start time count
    auto startTime = std::chrono::high_resolution_clock::now();

    // 3. Train based on batch size
    if (isBatchTrain == false) {
        batchSize = 1;
        int fileCount = 0;
        learningRate = 0.001f;
        std::cout << "Training with batch size: 1" << std::endl;
        for(const auto& filePath : filePaths) {
            // Skip files that have already been processed in previous sessions
            if (fileCount < this->mnn2dPrg.filesProcessed) {
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
            input = in;
            target = exp;
            #ifdef USE_CPU
                train(in, exp);
            #elif USE_CU
                cuTrain(in, exp);
            #elif USE_CL
                clTrain(in, exp);
            #endif

            // for progress tracking
            fileCount++;
            this->mnn2dPrg.filesProcessed++;
            filesInCurrentSession++;
            bool sessionEnd = 0;
            if (sessionFiles > 0 && filesInCurrentSession == this->mnn2dPrg.sessionSize) {
                std::cout << "Session file limit (" << this->mnn2dPrg.sessionSize << ") reached." << std::endl;
                auto endTime = std::chrono::high_resolution_clock::now();
                this->mnn2dPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
                this->mnn2dPrg.timeTakenForTraining = previousTrainingTime + this->mnn2dPrg.timeForCurrentSession;
                this->learningRate = this->mnn2dPrg.currentLearningRate;
                this->mnn2dPrg.totalSessionsOfTraining++;
                sessionEnd = 1;
            }
            std::cout<< "File count: " << fileCount << std::endl;

            // If a session size is defined and reached, stop training for this session
            if (sessionEnd == 1 || fileCount == totalFiles) {
                std::cout << "Processed " << fileCount << "/" << totalFiles << " files..." << std::endl;
                std::cout << "====== Diagnostics For Current Session ======" << std::endl;
                computeStats(cweights, bweights, cgradients, bgradients, activate);
                if (logProgressToCSV(this->mnn2dPrg, this->path2progress) == 1)
                    std::cout << "Progress logged successfully." << std::endl;
                else 
                    throw std::runtime_error("Failed to log progress to CSV: " + this->path2progress);
                serializeWeights(cweights, bweights, binFileAddress);
                std::cout << "============== To Next Session ==============" << std::endl;
                filesInCurrentSession = 0; // Reset for the next session
                if (fileCount == totalFiles) {
                    std::cout << "All files processed. Ending training." << std::endl;
                    this->mnn2dPrg.loss = this->mnn2dPrg.accLoss / static_cast<float>(this->mnn2dPrg.totalCycleCount);
                    break;
                }
            }
        }
    }
    else {
        batchSize = BATCH_SIZE;
        learningRate = 0.01f;
        std::cout << "Training With BatchSize of " << batchSize << std::endl;
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
        int startFileIndex = (this->mnn2dPrg.filesProcessed / batchSize) * batchSize;
        std::cout << "Starting from file index " << startFileIndex << " to align with batches." << std::endl;

        for(int i = startFileIndex; i < totalFiles; i += batchSize) {
            std::cout << "Processing from file index: " << i << " to " << i + batchSize << std::endl;
            std::vector<std::vector<std::vector<float>>> inBatch;
            std::vector<std::vector<float>> expBatch;
            int currentBatchEnd = std::min<int>(i + batchSize, totalFiles);

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

            inputBatch = inBatch;
            targetBatch = expBatch;
            #ifdef USE_CPU
                trainBatch(inputBatch, targetBatch);
            #elif USE_CU
                cuTrainBatch(inputBatch, targetBatch);
            #elif USE_CL
                clTrainBatch(inputBatch, targetBatch);
            #endif

            // for progress tracking
            fileCount += batchSize;
            this->mnn2dPrg.filesProcessed += batchSize;
            filesInCurrentSession += batchSize;
            bool sessionEnd = 0;
            if (sessionFiles > 0 && filesInCurrentSession == this->mnn2dPrg.sessionSize) {
                std::cout << "Session file limit (" << this->mnn2dPrg.sessionSize << ") reached." << std::endl;
                auto endTime = std::chrono::high_resolution_clock::now();
                this->mnn2dPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
                this->mnn2dPrg.timeTakenForTraining = previousTrainingTime + this->mnn2dPrg.timeForCurrentSession;
                this->learningRate = this->mnn2dPrg.currentLearningRate;
                this->mnn2dPrg.totalSessionsOfTraining++;
                sessionEnd = 1;
            }
            std::cout<< "File count: " << fileCount << std::endl;

            // If a session size is defined and reached, stop training for this session
            if (sessionEnd == 1 || fileCount == totalFiles) {
                std::cout << "Processed " << fileCount << "/" << totalFiles << " files..." << std::endl;
                std::cout << "====== Diagnostics For Current Session ======" << std::endl;
                computeStats(cweights, bweights, cgradients, bgradients, activate);
                if (logProgressToCSV(this->mnn2dPrg, this->path2progress) == 1)
                    std::cout << "Progress logged successfully." << std::endl;
                else 
                    throw std::runtime_error("Failed to log progress to CSV: " + this->path2progress);
                serializeWeights(cweights, bweights, binFileAddress);
                std::cout << "============== To Next Session ==============" << std::endl;
                filesInCurrentSession = 0; // Reset for the next session
                if (fileCount == totalFiles) {
                    std::cout << "All files processed. Ending training." << std::endl;
                    this->mnn2dPrg.loss = this->mnn2dPrg.accLoss / static_cast<float>(this->mnn2dPrg.totalCycleCount);
                    break;
                }
            }
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    this->mnn2dPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    this->mnn2dPrg.timeTakenForTraining = previousTrainingTime + this->mnn2dPrg.timeForCurrentSession;
    serializeWeights(cweights, bweights, binFileAddress);
    logProgressToCSV(this->mnn2dPrg, this->path2progress);
    std::cout << "--- Training Finished (mnn2d) ---" << std::endl;
}