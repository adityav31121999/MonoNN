#include <filesystem>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include "progress.hpp"

/**
 * @brief this is to store session-wise per epoch data for detailed analysis
 */
void sessionDataToCsv(const std::string& dir2Ses, int epoch, int session,
                bool batchOrNot, const std::vector<std::vector<float>> &weightStats,
                const std::vector<std::vector<int>> &confusion, const confMat &cm,
                const scores &sc, const progress &p)
{
    std::filesystem::path dirPath = dir2Ses;
    if (!dirPath.empty() && !std::filesystem::exists(dirPath)) {
        std::cout << "Directory " << dirPath << " does not exist. Creating it." << std::endl;
        if (!std::filesystem::create_directories(dirPath)) {
            throw std::runtime_error("MNN CONSTRUCTOR: could not create directory: " + dirPath.string());
        }
    }

    // file name = e_epoch_s_session_b_batchOrNot.csv
    std::string newCsv = dir2Ses + "/e" + std::to_string(epoch) + "_s_" + std::to_string(session) + "_b_" + std::to_string(batchOrNot) + ".csv";
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
    for (size_t i = 0; i < weightStats.size()/5; ++i) {
        file << "cweights" + i;
        for (const auto& stat : weightStats[i]) {
            file << "," << stat;
        }
        file << "\n";
    }
    for (size_t i = 0; i < weightStats.size()/5; ++i) {
        file << "bweights" + i;
        for (const auto& stat : weightStats[i]) {
            file << "," << stat;
        }
        file << "\n";
    }
    for (size_t i = 0; i < weightStats.size()/5; ++i) {
        file << "cgradients" + i;
        for (const auto& stat : weightStats[i]) {
            file << "," << stat;
        }
        file << "\n";
    }
    for (size_t i = 0; i < weightStats.size()/5; ++i) {
        file << "bgradients" + i;
        for (const auto& stat : weightStats[i]) {
            file << "," << stat;
        }
        file << "\n";
    }
    for (size_t i = 0; i < weightStats.size()/5; ++i) {
        file << "activations" + i;
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