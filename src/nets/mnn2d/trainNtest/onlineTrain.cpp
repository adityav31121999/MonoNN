#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include <random>
#include "mnn2d.hpp"

// for MNN2D

/**
 * @brief single epoch multi-cycle aggressive online training
 * @param dataSetPath path to dataset folder.
 * @param useThreadOrBuffer use threads in CPU and full buffer-based operation in CUDA and OpenCL
 * @param isRGB if image is RGB 1, else 0 for grey image
 */
void mnn2d::onlineTraining(const std::string &dataSetPath, bool isRGB, bool useThreadOrBuffer)
{
    std::cout << "This is onlineTraining.\n";
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
    std::cout << "\n--- Starting Training (mnn) ---" << std::endl;

    // access training progress information from file address and use it to re-start training
    // from new session
    double previousTrainingTime = 0.0;
    int epochCount = 1;
    int curPreds = 0;
    int fileCount = 0;
    allScores = {};
    confusion.assign(outSize, std::vector<int>(outSize, 0));
    if (!loadLastProgress(trainPrg, this->path2progress)) {
        // Preserve session and batch size set before calling train
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
        epochCount = trainPrg.epoch;
    }

    int filesInCurrentSession = 0;
    // Update progress struct with file and batch info
    trainPrg.totalTrainFiles = totalFiles;
    trainPrg.batchSize = batchSize;
    int sessionFiles = trainPrg.sessionSize; // * trainPrg.batchSize;
    std::cout << "Session Size (in batches/session): " << trainPrg.sessionSize << std::endl;
    std::cout << "Files in Single Session: " << sessionFiles << std::endl;
    std::cout << "learning rate: " << learningRate << std::endl;

    // start time count
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

            // Convert image to a 2D vector
            std::vector<std::vector<float>> in = image2matrix(filePath.string(), isRGB);
            // Extract label and create one-hot target vector
            std::string filename = filePath.stem().string();
            int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
            std::vector<float> exp(this->outSize, 0.0f);
            if (label < this->outSize) {
                exp[label] = 1.0f;
            }
            for(int i = 0; i < in.size(); i++) {
                for(int j = 0; j < in[i].size(); j++) {
                    in[i][j] /= 255;
                }
            }
            input = in;
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

            if (label < confusion.size() && maxIndex(output) < confusion[0].size()) {
                confusion[label][maxIndex(output)]++;
            }
            if(maxIndex(output) == maxIndex(exp))
                correctPredictions++;
            trainPrg.accLoss += crossEntropy(output, exp);
            getScore(output, exp, allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);

            bool sessionEnd = 0;
            if ((sessionFiles > 0 && filesInCurrentSession == sessionFiles) || fileCount == totalFiles) {
                auto endTime = std::chrono::high_resolution_clock::now();
                trainPrg.correctPredPercent = static_cast<float>(100 * correctPredictions) / fileCount;
                trainPrg.timeForCurrentSession = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
                trainPrg.timeTakenForTraining = previousTrainingTime + trainPrg.timeForCurrentSession;
                trainPrg.currentLearningRate = this->learningRate;
                trainPrg.sessionCount++;
                trainPrg.totalCycleCount += filesInCurrentSession;
                trainPrg.trainingPredictions = correctPredictions;
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
                std::cout << "Epoch: " << trainPrg.epoch << "\tFiles Processed: " << fileCount << "/" << totalFiles
                          << "\tPredictions: " << correctPredictions
                          << "\tTraining Accuracy: " << trainPrg.correctPredPercent << "%"
                          << "\tLoss: " << trainPrg.accLoss / static_cast<float>(trainPrg.filesProcessed)
                          << std::endl;
            }

            // If a session size is defined and reached, stop training for this session
            if (sessionEnd == 1 || fileCount == totalFiles) {
                if (fileCount == totalFiles) trainPrg.loss = trainPrg.accLoss / static_cast<float>(totalFiles);
                if (logProgressToCSV(trainPrg, this->path2progress) != 1) {
                    throw std::runtime_error("Failed to log progress to CSV: " + this->path2progress);
                }
                serializeWeights(cweights, bweights, binFileAddress);
                filesInCurrentSession = 0; // Reset for the next session
                if (fileCount == totalFiles) {
                    std::cout << "All files processed. Ending training for this epoch." << std::endl;
                    bool notBatch = false;
                    confData = {};
                    confData = confusionMatrixFunc(confusion);
                    allScores.sse = allScores.totalSumOfError / totalFiles;
                    allScores.ssr = allScores.totalSumOfRegression / totalFiles;
                    allScores.sst = allScores.totalSumOfSquares / totalFiles;
                    allScores.r2 = allScores.ssr / allScores.sst;
                    weightStats.clear();
                    computeStatsForCsv(cweights, bweights, weightStats);
                    epochDataToCsv1(dataSetPath + "/mnn2d/epoch", trainPrg.epoch, notBatch, weightStats, confusion, confData, allScores,
                                   trainPrg, 1);
                    std::string epochWeight = dataSetPath + "/mnn2d/epochWeights" + std::to_string(trainPrg.epoch) + ".bin";
                    createBinFile(epochWeight, param);
                    serializeWeights(cweights, bweights, epochWeight);
                    break;
                }
            }
        }

        fileCount = 0;
        trainPrg.filesProcessed = 0;
        if(trainPrg.correctPredPercent >= 97.0f) {
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

    std::cout << "--- Training Finished (mnn2d) ---" << std::endl;
}