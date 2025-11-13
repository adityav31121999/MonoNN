#include "mnn.hpp"
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include <random>

/**
 * @brief train network on given dataset
 * @param dataSetPath path to dataset folder
 * @param batchSize numebr of inputs in single batch (1 or more)
 */
void mnn::train(const std::string &dataSetPath, int batchSize)
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
        return;
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    // 2. Sort the dataset to ensure consistent order for resumable training
    std::sort(filePaths.begin(), filePaths.end());

    int totalFiles = filePaths.size();
    std::cout << "\n--- Starting Training (mnn) ---" << std::endl;
    std::cout << "Resuming from file index: " << this->mnnPrg.filesProcessed << std::endl;
    std::cout << "Found " << totalFiles << " files for training." << std::endl;

    this->mnnPrg.currentLearningRate = this->learningRate;
    // Update progress struct with file and batch info
    this->mnnPrg.totalTrainFiles = totalFiles;
    this->mnnPrg.batchSize = batchSize;
    // 3. Train based on batch size
    if (batchSize == 1) {
        int fileCount = 0;
        std::cout << "Training with batch size: 1" << std::endl;
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

            #ifdef USE_CPU
                train(in, exp);
            #elif USE_CUDA
                cuTrain(in, exp);
            #elif USE_CL
                clTrain(in, exp);
            #endif

            this->mnnPrg.filesProcessed++;
            fileCount++;

            if (fileCount % 100 == 0 || fileCount == totalFiles) {
                std::cout << "Processed " << fileCount << "/" << totalFiles << " files..." << std::endl;
            }
            // If a session size is defined and reached, stop training for this session
            if (this->mnnPrg.sessionSize > 0 && fileCount >= this->mnnPrg.sessionSize) {
                std::cout << "Session batch limit (" << this->mnnPrg.sessionSize << ") reached." << std::endl;
                auto endTime = std::chrono::high_resolution_clock::now();
                this->mnnPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
                this->learningRate = this->mnnPrg.currentLearningRate;
                logProgressToCSV(this->mnnPrg, "training_progress.csv");
                break;
            }
        }
    }
    else if (batchSize > 1) {
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

            #ifdef USE_CPU
                trainBatch(inBatch, expBatch);
            #elif USE_CUDA
                cuTrainBatch(inBatch, expBatch);
            #elif USE_CL
                clTrainBatch(inBatch, expBatch);
            #endif
            batchesProcessed++;
            this->mnnPrg.filesProcessed += inBatch.size();

            // If a session size is defined and reached, stop training for this session
            if (this->mnnPrg.sessionSize > 0 && batchesProcessed >= this->mnnPrg.sessionSize) {
                std::cout << "Session batch limit (" << this->mnnPrg.sessionSize << ") reached." << std::endl;
                break;
                auto endTime = std::chrono::high_resolution_clock::now();
                this->mnnPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
                this->learningRate = this->mnnPrg.currentLearningRate;
                serializeWeights(cweights, bweights, binFileAddress);
                logProgressToCSV(this->mnnPrg, "training_progress.csv");
                break;

            }
            if (batchesProcessed % 10 == 0 || batchesProcessed == totalBatches) {
                std::cout << "Processed batch " << batchesProcessed << "/" << totalBatches << "..." << std::endl;
            }
        }
    }
    else {
        throw std::runtime_error("Invalid batch size: " + std::to_string(batchSize));
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    this->mnnPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    serializeWeights(cweights, bweights, binFileAddress);
    logProgressToCSV(this->mnnPrg, "training_progress.csv");
    std::cout << "--- Training Finished (mnn) ---" << std::endl;
}

/**
 * @brief train network on given dataset
 * @param dataSetPath path to dataset folder.
 * @param batchSize number of inputs in a single batch (1 or more).
 */
void mnn2d::train(const std::string &dataSetPath, int batchSize)
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
        return;
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    // 2. Sort the dataset to ensure consistent order for resumable training
    std::sort(filePaths.begin(), filePaths.end());

    int totalFiles = filePaths.size();
    std::cout << "\n--- Starting Training (mnn2d) ---" << std::endl;
    std::cout << "Resuming from file index: " << this->mnn2dPrg.filesProcessed << std::endl;
    std::cout << "Found " << totalFiles << " files for training." << std::endl;

    this->mnn2dPrg.currentLearningRate = this->learningRate;
    // Update progress struct with file and batch info
    this->mnn2dPrg.totalTrainFiles = totalFiles;
    this->mnn2dPrg.batchSize = batchSize;

    // 3. Train based on batch size
    if (batchSize == 1) {
        int fileCount = 0;
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

            #ifdef USE_CPU
                train(in, exp);
            #elif USE_CUDA
                cuTrain(in, exp);
            #elif USE_CL
                clTrain(in, exp);
            #endif

            this->mnn2dPrg.filesProcessed++;
            fileCount++;

            if (fileCount % 100 == 0 || fileCount == totalFiles) {
                std::cout << "Processed " << fileCount << "/" << totalFiles << " files..." << std::endl;
            }
            // If a session size is defined and reached, stop training for this session
            if (this->mnn2dPrg.sessionSize > 0 && fileCount >= this->mnn2dPrg.sessionSize) {
                std::cout << "Session batch limit (" << this->mnn2dPrg.sessionSize << ") reached." << std::endl;
                auto endTime = std::chrono::high_resolution_clock::now();
                this->mnn2dPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
                this->learningRate = this->mnn2dPrg.currentLearningRate;
                logProgressToCSV(this->mnn2dPrg, "training_progress.csv");
                break;
            }
        }
    }
    else if (batchSize > 1) {
        unsigned int batchesProcessed = 0;
        int totalBatches = (totalFiles + batchSize - 1) / batchSize;
        std::cout << "Training with batch size: " << batchSize << " (" << totalBatches << " batches)" << std::endl;

        // Start iterating from the beginning of the batch where the last processed file was
        int startFileIndex = (this->mnn2dPrg.filesProcessed / batchSize) * batchSize;
        std::cout << "Starting from file index " << startFileIndex << " to align with batches." << std::endl;

        for(int i = startFileIndex; i < totalFiles; i += batchSize) {
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

            #ifdef USE_CPU
                trainBatch(inBatch, expBatch);
            #elif USE_CUDA
                cuTrainBatch(inBatch, expBatch);
            #elif USE_CL
                clTrainBatch(inBatch, expBatch);
            #endif
            batchesProcessed++;
            this->mnn2dPrg.filesProcessed += inBatch.size();

            // If a session size is defined and reached, stop training for this session
            if (this->mnn2dPrg.sessionSize > 0 && batchesProcessed >= this->mnn2dPrg.sessionSize) {
                std::cout << "Session batch limit (" << this->mnn2dPrg.sessionSize << ") reached." << std::endl;
                break;
                auto endTime = std::chrono::high_resolution_clock::now();
                this->mnn2dPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
                this->learningRate = this->mnn2dPrg.currentLearningRate;
                serializeWeights(cweights, bweights, binFileAddress);
                logProgressToCSV(this->mnn2dPrg, "training_progress.csv");
                break;

            }
            if (batchesProcessed % 10 == 0 || batchesProcessed == totalBatches) {
                std::cout << "Processed batch " << batchesProcessed << "/" << totalBatches << "..." << std::endl;
            }
        }
    }
    else {
        throw std::runtime_error("Invalid batch size: " + std::to_string(batchSize));
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    this->mnn2dPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    serializeWeights(cweights, bweights, binFileAddress);
    logProgressToCSV(this->mnn2dPrg, "training_progress.csv");
    std::cout << "--- Training Finished (mnn2d) ---" << std::endl;
}