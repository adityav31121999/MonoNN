#include "mnn.hpp"
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <vector>
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

    // 2. Shuffle the dataset for randomness
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(filePaths.begin(), filePaths.end(), g);

    int totalFiles = filePaths.size();
    std::cout << "\n--- Starting Training (mnn) ---" << std::endl;
    std::cout << "Found " << totalFiles << " files for training." << std::endl;

    // 3. Train based on batch size
    if (batchSize == 1) {
        int fileCount = 0;
        std::cout << "Training with batch size: 1" << std::endl;
        for(const auto& filePath : filePaths) {
            fileCount++;
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
            #elif USE_OPENCL
                clTrain(in, exp);
            #endif

            if (fileCount % 100 == 0 || fileCount == totalFiles) {
                std::cout << "Processed " << fileCount << "/" << totalFiles << " files..." << std::endl;
            }
        }
    }
    else if (batchSize > 1) {
        int batchesProcessed = 0;
        int totalBatches = (totalFiles + batchSize - 1) / batchSize;
        std::cout << "Training with batch size: " << batchSize << " (" << totalBatches << " batches)" << std::endl;
        for(int i = 0; i < totalFiles; i += batchSize) {
            std::vector<std::vector<float>> inBatch;
            std::vector<std::vector<float>> expBatch;
            int currentBatchEnd = std::min(i + batchSize, totalFiles);

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
            #elif USE_OPENCL
                clTrainBatch(inBatch, expBatch);
            #endif
            batchesProcessed++;
            if (batchesProcessed % 10 == 0 || batchesProcessed == totalBatches) {
                std::cout << "Processed batch " << batchesProcessed << "/" << totalBatches << "..." << std::endl;
            }
        }
    }
    else {
        throw std::runtime_error("Invalid batch size: " + std::to_string(batchSize));
    }
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

    // 2. Shuffle the dataset for randomness
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(filePaths.begin(), filePaths.end(), g);

    int totalFiles = filePaths.size();
    std::cout << "\n--- Starting Training (mnn2d) ---" << std::endl;
    std::cout << "Found " << totalFiles << " files for training." << std::endl;

    // 3. Train based on batch size
    if (batchSize == 1) {
        int fileCount = 0;
        std::cout << "Training with batch size: 1" << std::endl;
        for(const auto& filePath : filePaths) {
            fileCount++;
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
            #elif USE_OPENCL
                clTrain(in, exp);
            #endif

            if (fileCount % 100 == 0 || fileCount == totalFiles) {
                std::cout << "Processed " << fileCount << "/" << totalFiles << " files..." << std::endl;
            }
        }
    }
    else if (batchSize > 1) {
        int batchesProcessed = 0;
        int totalBatches = (totalFiles + batchSize - 1) / batchSize;
        std::cout << "Training with batch size: " << batchSize << " (" << totalBatches << " batches)" << std::endl;
        for(int i = 0; i < totalFiles; i += batchSize) {
            std::vector<std::vector<std::vector<float>>> inBatch;
            std::vector<std::vector<float>> expBatch;
            int currentBatchEnd = std::min(i + batchSize, totalFiles);

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
            #elif USE_OPENCL
                clTrainBatch(inBatch, expBatch);
            #endif
            batchesProcessed++;
            if (batchesProcessed % 10 == 0 || batchesProcessed == totalBatches) {
                std::cout << "Processed batch " << batchesProcessed << "/" << totalBatches << "..." << std::endl;
            }
        }
    }
    else {
        throw std::runtime_error("Invalid batch size: " + std::to_string(batchSize));
    }
    std::cout << "--- Training Finished (mnn2d) ---" << std::endl;
}