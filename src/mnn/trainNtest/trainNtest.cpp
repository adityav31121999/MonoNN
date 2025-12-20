#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include "mnn.hpp"
#include "mnn2d.hpp"

// for MNN

/**
 * @brief train and test mnn neural network
 * @param dataSetPath path to training and testing data set
 * @param useThreadOrBuffer use thread/buffer based functions or not
 */
void mnn::trainNtest(const std::string &dataSetPath, bool useThreadOrBuffer)
{
    // Check if dataSetPath/mnn1d/traintest.csv exists or not. If not, create it.
    std::string trainTestCsvPath = dataSetPath + "/mnn1d/traintest.csv";
    try {
        if (!std::filesystem::exists(trainTestCsvPath)) {
            std::filesystem::create_directories(dataSetPath + "/mnn1d");
            std::ofstream ofs(trainTestCsvPath);
            if (!ofs.is_open()) {
                throw std::runtime_error("Failed to create traintest.csv at: " + trainTestCsvPath);
            }
            ofs << "pre-train, 1, 0\n";
            ofs << "train, 2, 0\n";
            ofs << "test, 3, 0\n";
            ofs.close();
        }
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Filesystem error when checking/creating traintest.csv: " + std::string(e.what()));
    }

    auto stages = readTrainTestCsv(trainTestCsvPath);
    if (stages.empty()) {
        stages = {{"pre-train", 1, 0}, {"train", 2, 0}, {"test", 3, 0}};
        writeTrainTestCsv(trainTestCsvPath, stages);
    }

    for (auto& stage : stages) {
        if (stage.status == 1) continue;

        if (stage.id == 1) {
            preTrainRun(dataSetPath);
        } else if (stage.id == 2) {
            fullDataSetTraining(dataSetPath, useThreadOrBuffer);
        } else if (stage.id == 3) {
            test(dataSetPath, useThreadOrBuffer);
        }
        stage.status = 1;
        writeTrainTestCsv(trainTestCsvPath, stages);
    }
    std::cout << "--- Training and Testing Finished (mnn) ---" << std::endl;
}

// for MNN2D

/**
 * @brief train and test mnn2d neural network
 * @param dataSetPath path to training and testing data set
 * @param useThreadOrBuffer use thread/buffer based functions or not
 */
void mnn2d::trainNtest(const std::string &dataSetPath, bool useThreadOrBuffer)
{
    // Check if dataSetPath/mnn2d/traintest.csv exists or not. If not, create it.
    std::string trainTestCsvPath = dataSetPath + "/mnn2d/traintest.csv";
    try {
        if (!std::filesystem::exists(trainTestCsvPath)) {
            std::filesystem::create_directories(dataSetPath + "/mnn2d");
            std::ofstream ofs(trainTestCsvPath);
            if (!ofs.is_open()) {
                throw std::runtime_error("Failed to create traintest.csv at: " + trainTestCsvPath);
            }
            ofs << "pre-train, 1, 0\n";
            ofs << "train, 2, 0\n";
            ofs << "test, 3, 0\n";
            ofs.close();
        }
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Filesystem error when checking/creating traintest.csv: " + std::string(e.what()));
    }

    auto stages = readTrainTestCsv(trainTestCsvPath);
    if (stages.empty()) {
        stages = {{"pre-train", 1, 0}, {"train", 2, 0}, {"test", 3, 0}};
        writeTrainTestCsv(trainTestCsvPath, stages);
    }

    for (auto& stage : stages) {
        if (stage.status == 1) continue;

        if (stage.id == 1) {
            preTrainRun(dataSetPath);
        } else if (stage.id == 2) {
            fullDataSetTraining(dataSetPath, useThreadOrBuffer);
        } else if (stage.id == 3) {
            test(dataSetPath, useThreadOrBuffer);
        }
        stage.status = 1;
        writeTrainTestCsv(trainTestCsvPath, stages);
    }
    std::cout << "--- Training and Testing Finished (mnn) ---" << std::endl;
}