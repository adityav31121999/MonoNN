#include <filesystem>
#include <fstream>
#include <iostream>
#include "progress.hpp"

/**
 * @brief save pre-train, training session and epochs data to csv
 * @param path2dir address of epoch directory
 * @param epoch training cycle count
 * @param batchOrNot batch training or not
 * @param weightStats weight statistics (mean, dev, min, max)
 * @param confusion confusion matrix of epoch
 * @param cm confusion matrix related scores
 * @param sc coefficient of determination
 * @param p training progress
 * @param isTrainOrPre if is training == 1 else 0 for pre-train run
 */
void epochDataToCsv(const std::string &path2dir, const int epoch, bool batchOrNot,
                    const std::vector<std::vector<float>> &weightStats,
                    const std::vector<std::vector<int>> &confusion,
                    const confMat &cm, const scores &sc, const progress &p,
                    bool isTrainOrPre)
{
    std::filesystem::path dirPath = path2dir;
    if (!dirPath.empty() && !std::filesystem::exists(dirPath)) {
        std::cout << "Directory " << dirPath << " does not exist. Creating it." << std::endl;
        if (!std::filesystem::create_directories(dirPath)) {
            throw std::runtime_error("MNN CONSTRUCTOR: could not create directory: " + dirPath.string());
        }
    }

    std::string newCsv;
    if (isTrainOrPre == 1) {
        newCsv = path2dir + "/epoch" + std::to_string(epoch) + "_" + std::to_string(batchOrNot) + ".csv";
    }
    else {
        newCsv = path2dir + "/preTrainEpoch0.csv";
    }
    std::filesystem::create_directories(std::filesystem::path(newCsv).parent_path());
    std::ofstream file(newCsv);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << newCsv << std::endl;
        return;
    }

    // --- Progress ---
    file << "Progress\n";
    file << "epoch," << p.epoch << "\n";
    file << "batchSize," << p.batchSize << "\n";
    file << "sessionSize," << p.sessionSize << "\n";
    file << "totalTrainFiles," << p.totalTrainFiles << "\n";
    file << "filesProcessed," << p.filesProcessed << "\n";
    file << "currentLearningRate," << p.currentLearningRate << "\n";
    file << "loss," << p.loss << "\n";
    file << "accLoss," << p.accLoss << "\n";
    file << "trainingPredictions," << p.trainingPredictions << "\n";
    file << "correctPredPercent," << p.correctPredPercent << "\n";
    file << "totalCycleCount," << p.totalCycleCount << "\n";
    file << "sessionCount," << p.sessionCount << "\n";
    file << "timeForCurrentSession," << p.timeForCurrentSession << "\n";
    file << "timeTakenForTraining," << p.timeTakenForTraining << "\n";

    // --- Weight Stats ---
    file << "\nWeightStats\n";
    file << "property,layer,mean,std,min,max\n";
    size_t w = weightStats.size() / 4;
    for (size_t i = 0; i < w; ++i) {
        file << "cweights" + std::to_string(i);
        for (const auto& stat : weightStats[i]) {
            file << "," << stat;
        }
        file << "\n";
    }
    for (size_t i = 0; i < w; ++i) {
        file << "bweights" + std::to_string(i);
        for (const auto& stat : weightStats[w + i]) {
            file << "," << stat;
        }
        file << "\n";
    }
    for (size_t i = 0; i < w; ++i) {
        file << "cgradients" + std::to_string(i);
        for (const auto& stat : weightStats[(2 * w) + i]) {
            file << "," << stat;
        }
        file << "\n";
    }
    for (size_t i = 0; i < w; ++i) {
        file << "bgradients" + std::to_string(i);
        for (const auto& stat : weightStats[(3 * w) + i]) {
            file << "," << stat;
        }
        file << "\n";
    }

    // --- Confusion Matrix ---
    file << "\nConfusionMatrix\n";
    for (const auto& row : confusion) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << row[j] << (j == row.size() - 1 ? "" : ",");
        }
        file << "\n";
    }

    // --- Classification Metrics (confMat) ---
    file << "\nClassificationMetrics\n";
    file << "avgAccuracy," << cm.avgAccuracy << "\n";
    file << "macro_f1Score," << cm.macro_f1Score << "\n";
    file << "weighted_f1Score," << cm.weighted_f1Score << "\n";
    file << "class,accuracy,precision,recall,f1,support\n";
    for (size_t i = 0; i < cm.precision.size(); ++i) {
        file << i << "," << cm.accuracy[i] << "," << cm.precision[i] << "," << cm.recall[i] << "," << cm.f1[i] << "," << cm.support[i] << "\n";
    }

    // --- Regression Scores (scores) ---
    file << "\nRegressionScores\n";
    file << "r2," << sc.r2 << "\n";
    file << "sst," << sc.sst << "\n";
    file << "ssr," << sc.ssr << "\n";
    file << "sse," << sc.sse << "\n";

    file.close();
}


/**
 * @brief save pre-train run and each training epochs data to csv
 * @param path2dir address of epoch directory
 * @param epoch training cycle count
 * @param batchOrNot batch training or not
 * @param weightStats weight statistics (mean, dev, min, max) (excluding gradients)
 * @param confusion confusion matrix of epoch
 * @param cm confusion matrix related scores
 * @param sc coefficient of determination
 * @param p training progress
 * @param isTrainOrPre if is training == 1 else 0 for pre-train run
 */
void epochDataToCsv1(const std::string &path2dir, const int epoch, bool batchOrNot,
                    const std::vector<std::vector<float>> &weightStats,
                    const std::vector<std::vector<int>> &confusion,
                    const confMat &cm, const scores &sc, const progress &p,
                    bool isTrainOrPre)
{
    std::filesystem::path dirPath = path2dir;
    if (!dirPath.empty() && !std::filesystem::exists(dirPath)) {
        std::cout << "Directory " << dirPath << " does not exist. Creating it." << std::endl;
        if (!std::filesystem::create_directories(dirPath)) {
            throw std::runtime_error("MNN CONSTRUCTOR: could not create directory: " + dirPath.string());
        }
    }

    std::string newCsv;
    if (isTrainOrPre == 1) {
        newCsv = path2dir + "/epoch" + std::to_string(epoch) + "_" + std::to_string(batchOrNot) + ".csv";
    }
    else {
        newCsv = path2dir + "/preTrainEpoch0.csv";
    }
    std::filesystem::create_directories(std::filesystem::path(newCsv).parent_path());
    std::ofstream file(newCsv);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << newCsv << std::endl;
        return;
    }

    // --- Progress ---
    file << "Progress\n";
    file << "epoch," << p.epoch << "\n";
    file << "batchSize," << p.batchSize << "\n";
    file << "sessionSize," << p.sessionSize << "\n";
    file << "totalTrainFiles," << p.totalTrainFiles << "\n";
    file << "filesProcessed," << p.filesProcessed << "\n";
    file << "currentLearningRate," << p.currentLearningRate << "\n";
    file << "loss," << p.loss << "\n";
    file << "accLoss," << p.accLoss << "\n";
    file << "trainingPredictions," << p.trainingPredictions << "\n";
    file << "correctPredPercent," << p.correctPredPercent << "\n";
    file << "totalCycleCount," << p.totalCycleCount << "\n";
    file << "sessionCount," << p.sessionCount << "\n";
    file << "timeForCurrentSession," << p.timeForCurrentSession << "\n";
    file << "timeTakenForTraining," << p.timeTakenForTraining << "\n";

    // --- Weight Stats ---
    file << "\nWeightStats\n";
    file << "property,layer,mean,std,min,max\n";
    size_t w = weightStats.size() / 4;
    for (size_t i = 0; i < w; ++i) {
        file << "cweights" + std::to_string(i);
        for (const auto& stat : weightStats[i]) {
            file << "," << stat;
        }
        file << "\n";
    }
    for (size_t i = 0; i < w; ++i) {
        file << "bweights" + std::to_string(i);
        for (const auto& stat : weightStats[w + i]) {
            file << "," << stat;
        }
        file << "\n";
    }

    // --- Confusion Matrix ---
    file << "\nConfusionMatrix\n";
    for (const auto& row : confusion) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << row[j] << (j == row.size() - 1 ? "" : ",");
        }
        file << "\n";
    }

    // --- Classification Metrics (confMat) ---
    file << "\nClassificationMetrics\n";
    file << "avgAccuracy," << cm.avgAccuracy << "\n";
    file << "macro_f1Score," << cm.macro_f1Score << "\n";
    file << "weighted_f1Score," << cm.weighted_f1Score << "\n";
    file << "class,accuracy,precision,recall,f1,support\n";
    for (size_t i = 0; i < cm.precision.size(); ++i) {
        file << i << "," << cm.accuracy[i] << "," << cm.precision[i] << "," << cm.recall[i] << "," << cm.f1[i] << "," << cm.support[i] << "\n";
    }

    // --- Regression Scores (scores) ---
    file << "\nRegressionScores\n";
    file << "r2," << sc.r2 << "\n";
    file << "sst," << sc.sst << "\n";
    file << "ssr," << sc.ssr << "\n";
    file << "sse," << sc.sse << "\n";

    file.close();
}


/**
 * @brief save pre-train run and test data to csv
 * @param path2dir address of epoch directory
 * @param weightStats weight statistics (mean, dev, min, max)
 * @param confusion confusion matrix of epoch
 * @param cm confusion matrix related scores
 * @param sc coefficient of determination
 * @param p training progress
 * @param isTestOrPre if is test run == 1 else 0 for pre-train run
 */
void epochDataToCsv(const std::string& path2dir, const std::vector<std::vector<int>>& confusion,
					const confMat& cm, const scores& sc, const test_progress& p, bool isTestOrPre)
{
    std::filesystem::path dirPath = path2dir;
    if (!dirPath.empty() && !std::filesystem::exists(dirPath)) {
        std::cout << "Directory " << dirPath << " does not exist. Creating it." << std::endl;
        if (!std::filesystem::create_directories(dirPath)) {
            throw std::runtime_error("MNN CONSTRUCTOR: could not create directory: " + dirPath.string());
        }
    }

    std::string newCsv = path2dir + ((isTestOrPre == 1) ? "/test.csv" : "/preTest.csv");
    std::filesystem::create_directories(std::filesystem::path(newCsv).parent_path());
    std::ofstream file(newCsv);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << newCsv << std::endl;
        return;
    }

    // --- Test Progress ---
    file << "TestProgress\n";
    file << "totalTestFiles," << p.totalTestFiles << "\n";
    file << "testError," << p.testError << "\n";
    file << "testAccuracy," << p.testAccuracy << "\n";
    file << "correctPredictions," << p.correctPredictions << "\n";

    // --- Confusion Matrix ---
    file << "\nConfusionMatrix\n";
    for (const auto& row : confusion) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << row[j] << (j == row.size() - 1 ? "" : ",");
        }
        file << "\n";
    }

    // --- Classification Metrics (confMat) ---
    file << "\nClassificationMetrics\n";
    file << "avgAccuracy," << cm.avgAccuracy << "\n";
    file << "macro_f1Score," << cm.macro_f1Score << "\n";
    file << "weighted_f1Score," << cm.weighted_f1Score << "\n";
    file << "class,accuracy,precision,recall,f1,support\n";
    for (size_t i = 0; i < cm.precision.size(); ++i) {
        file << i << "," << cm.accuracy[i] << "," << cm.precision[i] << "," << cm.recall[i] << "," << cm.f1[i] << "," << cm.support[i] << "\n";
    }

    // --- Regression Scores (scores) ---
    file << "\nRegressionScores\n";
    file << "r2," << sc.r2 << "\n";
    file << "sst," << sc.sst << "\n";
    file << "ssr," << sc.ssr << "\n";
    file << "sse," << sc.sse << "\n";

    file.close();
}