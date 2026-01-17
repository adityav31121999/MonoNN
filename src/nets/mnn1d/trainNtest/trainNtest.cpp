#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>
#include <sstream>
#include "mnn1d.hpp"

// for MNN

/**
 * @brief train and test mnn1d neural network
 * @param dataSetPath path to training and testing data set
 * @param useThreadOrBuffer use thread/buffer based functions or not
 * @param isRGB if image is RGB 1, else 0 for grey image
 * @param typeOfTrain 0 for fullDataSetTraining, 1 for onlineTraining, 2 for miniBatchTraining
 */
void mnn1d::trainNtest(const std::string &dataSetPath, bool isRGB, bool useThreadOrBuffer, int typeOfTrain)
{
    // Check if dataSetPath/mnn1d/traintest.csv exists or not. If not, create it.
    std::string trainTestCsvPath = dataSetPath + "/mnn1d/traintest.csv";
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

    std::cout << "Training files: " << std::endl;
    printClassDistribution(filePaths, outSize);

    std::vector<std::filesystem::path> filePathstest;
    std::string testPath = dataSetPath + "/test";

    try {
        for (const auto& entry : std::filesystem::directory_iterator(testPath)) {
            if (entry.is_regular_file()) {
                filePathstest.push_back(entry.path());
            }
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Failed to read dataset directory: " + std::string(e.what()));
    }

    if (filePathstest.empty()) {
        std::cout << "Warning: No files found in dataset directory: " << testPath << std::endl;
        return;
    }

    std::cout << "Training files: " << std::endl;
    printClassDistribution(filePathstest, outSize);

    try {
        if (!std::filesystem::exists(trainTestCsvPath)) {
            std::filesystem::create_directories(dataSetPath + "/mnn");
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
            preTrainRun(dataSetPath, isRGB);
        }
        else if (stage.id == 2) {
            std::cout << "Pre-training run completed in previous stage." << std::endl;
            if (typeOfTrain == 0) fullDataSetTraining(dataSetPath, isRGB, useThreadOrBuffer);
            else if (typeOfTrain == 1) onlineTraining(dataSetPath, isRGB, useThreadOrBuffer);
            else if (typeOfTrain == 2) miniBatchTraining(dataSetPath, isRGB, useThreadOrBuffer);
            else
                throw std::runtime_error("Invalid typeOfTrain value: " + std::to_string(typeOfTrain));
        }
        else if (stage.id == 3) {
            std::cout << "Pre-training run completed in previous stage." << std::endl;
            std::cout << "Training completed in previous stage." << std::endl;
            test(dataSetPath, isRGB, useThreadOrBuffer);
        }
        stage.status = 1;
        writeTrainTestCsv(trainTestCsvPath, stages);
    }
    std::cout << "--- Training and Testing Finished (mnn1d) ---" << std::endl;
}