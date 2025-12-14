#include <filesystem>
#include <fstream>
#include <iostream>
#include "progress.hpp"

/**
 * @brief save each training epochs data to csv
 * @param dataSetAddress address of dataset
 * @param epoch training cycle count
 * @param batchOrNot batch training or not
 * @param weightStats weight statistics (mean, dev, min, max)
 * @param confusion confusion matrix of epoch
 * @param cm confusion matrix related scores
 * @param sc coefficient of determination
 * @param p training progress
 */
void epochDataToCsv(const std::string &dataSetAddress, const int epoch, bool batchOrNot,
                    const std::vector<std::vector<float>> &weightStats,
                    const std::vector<std::vector<int>> &confusion,
                    const confMat &cm, const scores &sc, const progress &p)
{
    std::string newCsv = dataSetAddress + "/epoch" + std::to_string(epoch) + "_" + std::to_string(batchOrNot) + ".csv";
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
    file << "totalSessionsOfTraining," << p.totalSessionsOfTraining << "\n";
    file << "timeForCurrentSession," << p.timeForCurrentSession << "\n";
    file << "timeTakenForTraining," << p.timeTakenForTraining << "\n";

    // --- Weight Stats ---
    file << "\nWeightStats\n";
    file << "layer,mean,std,min,max\n";
    for (size_t i = 0; i < weightStats.size(); ++i) {
        file << i;
        for (const auto& stat : weightStats[i]) {
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
 * @brief save each training epochs data to csv
 * @param dataSetAddress address of dataset
 * @param weightStats weight statistics (mean, dev, min, max)
 * @param confusion confusion matrix of epoch
 * @param cm confusion matrix related scores
 * @param sc coefficient of determination
 * @param p training progress
 */
void epochDataToCsv(const std::string& dataSetAddress,
                    const std::vector<std::vector<int>>& confusion,
					const confMat& cm, const scores& sc, const test_progress& p)
{
    std::string newCsv = dataSetAddress + "/test.csv";
    std::filesystem::create_directories(std::filesystem::path(newCsv).parent_path());
    std::ofstream file(newCsv);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << newCsv << std::endl;
        return;
    }

    // --- Test Progress ---
    file << "TestProgress\n";
    file << "totalTestFiles," << p.totalTestFiles << "\n";
    file << "testFilesProcessed," << p.testFilesProcessed << "\n";
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