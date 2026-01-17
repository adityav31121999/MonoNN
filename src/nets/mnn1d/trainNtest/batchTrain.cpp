#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include <random>
#include <fstream>
#include "mnn1d.hpp"
// for MNN

/**
 * @brief single cycle multi epoch mini-batch online training
 * @param dataSetPath path to dataset folder
 * @param useThreadOrBuffer use threads in CPU and full buffer-based operation in CUDA and OpenCL
 * @param isRGB if image is RGB 1, else 0 for grey image
 */
void mnn1d::miniBatchTraining(const std::string &dataSetPath, bool isRGB, bool useThreadOrBuffer)
{
    std::cout << "This is miniBatchTraining.\n";
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

    // Sort the dataset to ensure consistent order for resumable training
    std::sort(filePaths.begin(), filePaths.end());

    int totalFiles = filePaths.size();
    std::cout << "\n--- Starting Mini-Batch Training (mnn1d) ---" << std::endl;
    std::cout << "Resuming from file index: " << trainPrg.filesProcessed << std::endl;

    std::string synopsisFilePath = dataSetPath + "/mnn1d/epochSynopsis.csv";
    std::ofstream synopsisFile;
    if (std::filesystem::exists(synopsisFilePath)) {
        synopsisFile.open(synopsisFilePath, std::ios::app);
    } else {
        std::filesystem::create_directories(dataSetPath + "/mnn1d");
        synopsisFile.open(synopsisFilePath);
        if (synopsisFile.is_open()) {
            synopsisFile << "Epoch,CorrectPredictions,Loss,Accuracy\n";
        }
    }

    // access training progress information from file address and use it to re-start training
    // from new session
    double previousTrainingTime = 0.0;
    int epochCount = 1;
    int curPreds = 0;
    int fileCount = 0;
    batchSize = BATCH_SIZE;
    int totalBatches = (totalFiles + batchSize - 1) / batchSize;
    if (!loadLastProgress(trainPrg, this->path2progress)) {
        std::cout << "No progress file found or file is empty. Starting fresh training." << std::endl;
        trainPrg = {}; // Reset progress
        confData = {};
        // trainPrg.sessionSize = ((totalFiles % SESSION_SIZE == 0) && (totalFiles / SESSION_SIZE <= 1000)) ? SESSION_SIZE : (totalFiles / 100);
        trainPrg.sessionSize = SESSION_SIZE;
        trainPrg.batchSize = BATCH_SIZE;
        trainPrg.currentLearningRate = this->learningRate;
        curPreds = 0;
        fileCount = 0;
        trainPrg.filesProcessed = 0;
    }
    else {
        std::cout << "Successfully loaded progress." << std::endl;
        if(trainPrg.filesProcessed == 0)
            std::cout << "Fresh training starts!" << std::endl;
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
        epochCount = trainPrg.epoch + 1;
    }

    int filesInCurrentSession = 0;
    // Update progress struct with file and batch info
    trainPrg.totalTrainFiles = totalFiles;
    trainPrg.batchSize = batchSize;
    int sessionFiles = trainPrg.sessionSize * trainPrg.batchSize;
    std::cout << "Batch Training with batch size: " << BATCH_SIZE << std::endl;
    std::cout << "Session Size (in batches/session): " << trainPrg.sessionSize << std::endl;
    std::cout << "Files in Single Session: " << sessionFiles << std::endl;

    // start time count
    auto startTime = std::chrono::high_resolution_clock::now();

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
    unsigned int batchesProcessed = 0;
    std::cout << "Training with batch size: " << batchSize << " (" << totalBatches << " batches)" << std::endl;

    // Start iterating from the beginning of the batch where the last processed file was
    int startFileIndex = (trainPrg.filesProcessed / batchSize) * batchSize;
    fileCount = startFileIndex;
    std::cout << "Starting from file index " << startFileIndex << " to align with batches." << std::endl;

    while (1) {
        // Shuffle filePaths at the beginning of each epoch
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(filePaths.begin(), filePaths.end(), g);
        // Resize confusion matrix before use
        confusion.assign(outSize, std::vector<int>(outSize, 0));

        unsigned int correctPredictions = curPreds;
        for(int i = startFileIndex; i < totalFiles; i += batchSize) {
            std::vector<std::vector<float>> inBatch;
            std::vector<std::vector<float>> expBatch;
            int currentBatchEnd = std::min<int>(i + batchSize, totalFiles);
            // get image
            for(int j = i; j < currentBatchEnd; ++j) {
                const auto& filePath = filePaths[j];
                inBatch.push_back(flatten(image2matrix(filePath.string(), isRGB)));
                std::string filename = filePath.stem().string();
                int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
                std::vector<float> exp(this->outSize, 0.0f);
                if (label < this->outSize) {
                    exp[label] = 1.0f;
                }
                expBatch.push_back(exp);
            }
            for(auto& img : inBatch) {
                if (std::any_of(img.begin(), img.end(), [](float val) { return val > 1.0f; })) {
                    std::transform(img.begin(), img.end(), img.begin(), [](float val) { return val / 255.0f; });
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
            fileCount += batchSize;
            trainPrg.filesProcessed += batchSize;
            filesInCurrentSession += batchSize;

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
        if(trainPrg.correctPredPercent >= 98.0f) {
            std::cout << "Training completed using minibatch of size " << BATCH_SIZE 
                      << "with accuracy of " << trainPrg.correctPredPercent << "% after "
                      << trainPrg.epoch << " epochs." << std::endl;
            break;
        }

        if (epochCount >= 4 || trainPrg.correctPredPercent >= 90.0f) {
            learningRate = cosineAnnealing(LEARNING_MAX, LEARNING_MIN, epochCount, EPOCH);
            std::cout << "New Learning Rate For Epoch " << epochCount << " : " << learningRate << std::endl;
        }
        epochCount++;
        if (epochCount == EPOCH) {
            std::cout << "Training Completed at accuracy " << trainPrg.correctPredPercent << "% at epoch " << trainPrg.epoch << "." << std::endl;
            break;
        }

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
        trainPrg.sessionCount= 0;
        trainPrg.timeForCurrentSession = 0;
        logProgressToCSV(trainPrg, this->path2progress);
        correctPredictions = 0;
        allScores = {};
        confData = {};
        confusion.clear();
        confusion.assign(outSize, std::vector<int>(outSize, 0));
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    trainPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    trainPrg.timeTakenForTraining = previousTrainingTime + trainPrg.timeForCurrentSession;
    serializeWeights(cweights, bweights, binFileAddress);
    logProgressToCSV(trainPrg, this->path2progress);
    std::cout << "--- Training Finished (mnn1d) ---" << std::endl;
}