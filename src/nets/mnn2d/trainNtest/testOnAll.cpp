#include <vector>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include "mnn2d.hpp"

// for MNN2D

/**
 * @brief record performance of network on all weights obtained on each epochs using test files
 * @param dataSetPath path to dataset
 * @param isRGB is file/image color or grey
 * @param useThreadOrBuffer 
 */
void mnn2d::testOnAllWeights(std::string dataSetPath, bool isRGB, bool useThreadOrBuffer)
{
    // first find all .bin files in the path
    int totalBinFiles = 0;
    std::vector<std::string> binFiles;
    std::string weightsPath = dataSetPath + "/mnn2d";
    for (const auto& file_entry : std::filesystem::directory_iterator(weightsPath)) {
        if (file_entry.is_regular_file() && file_entry.path().extension() == ".bin") {
            binFiles.push_back(file_entry.path().string());
        }
    }
    if (binFiles.empty()) {
        std::cout << "Warning: No .bin weight files found in " << weightsPath << std::endl;
        return;
    }

    // Sort bin files to a specific order: initialised, epochs, then trained.
    auto get_sort_key = [](const std::string& filename) -> std::pair<int, int> {
        std::filesystem::path path(filename);
        std::string name = path.stem().string();

        if (name == "initialisedWeights") {
            return {0, 0};
        }
        if (name.rfind("epochWeights", 0) == 0) {
            try {
                // "epochWeights" is 12 chars
                return {1, std::stoi(name.substr(12))};
            } catch (const std::exception&) {
                return {2, 0}; // Fallback for malformed names
            }
        }
        if (name == "trainedWeights") {
            return {3, 0};
        }
        return {2, 0}; // Other files
    };

    std::sort(binFiles.begin(), binFiles.end(), [&](const std::string& a, const std::string& b) {
        auto keyA = get_sort_key(a);
        auto keyB = get_sort_key(b);
        if (keyA.first != keyB.first) return keyA.first < keyB.first;
        if (keyA.first == 1) return keyA.second < keyB.second; // Sort epochs by number
        return a < b; // Lexicographical for others in the same group
    });

    // 1. Access all image files from the dataset path
    std::vector<std::filesystem::path> filePaths;
    std::string testPath = dataSetPath + "/test";
    try {
        for (const auto& entry : std::filesystem::directory_iterator(testPath)) {
            if (entry.is_regular_file()) {
                filePaths.push_back(entry.path());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Failed to read dataset directory: " + testPath + ": " + std::string(e.what()));
    }

    if (filePaths.empty()) {
        std::cout << "Warning: No files found in dataset directory: " << testPath << std::endl;
        testPrg.testError = 0.0f;
        return;
    }

    std::string path2all = dataSetPath + "/mnn2d/allTests";
    try {
        if (!std::filesystem::exists(path2all)) {
            std::filesystem::create_directories(path2all);
        }
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Filesystem error when checking/creating allTests directory: " + std::string(e.what()));
    }

    unsigned int totalInputs = filePaths.size();

    std::cout << "\n--- Starting All Weights Test (mnn2d) ---" << std::endl;
    std::cout << "Found " << totalInputs << " files for testing." << std::endl;
    testPrg.totalTestFiles = totalInputs;

    std::vector<int> w(cweights.size()), h(bweights.size());
    for(int i = 0; i < w.size(); i++) h[i] = cweights[i].size();
    for(int i = 0; i < w.size(); i++) w[i] = cweights[i][0].size();

    std::string synopsisFilePath = dataSetPath + "/mnn2d/allTests/allWeightsTestSynopsis.csv";
    int processedCount = 0;
    if (std::filesystem::exists(synopsisFilePath)) {
        std::ifstream inFile(synopsisFilePath);
        std::string line;
        while (std::getline(inFile, line)) {
            if (!line.empty()) processedCount++;
        }
        if (processedCount > 0) processedCount--; // Exclude header
    }

    std::ofstream synopsisFile;
    if (processedCount > 0) {
        synopsisFile.open(synopsisFilePath, std::ios::app);
        std::cout << "Resuming from " << processedCount << " processed files." << std::endl;
    } else {
        synopsisFile.open(synopsisFilePath);
        if (synopsisFile.is_open()) {
            synopsisFile << "WeightFile,CorrectPredictions,AverageLoss,Accuracy\n";
        }
    }

    if (!synopsisFile.is_open()) {
        throw std::runtime_error("Failed to open synopsis file for writing: " + synopsisFilePath);
    }

    int fileCounter = 0;
    // then for each directory run this code in loop
    for (const auto& entry : binFiles) {
        if (fileCounter < processedCount) {
            fileCounter++;
            continue;
        }
        fileCounter++;

        // Reset counters for each weight file
        unsigned int correctPredictions = 0;
        float accLoss = 0.0f;
        allScores.totalSumOfSquares = 0.0;
        allScores.totalSumOfRegression = 0.0;
        allScores.totalSumOfError = 0.0;
        confusion.assign(outSize, std::vector<int>(outSize, 0));
        std::cout << "\n--- Testing weights: " << entry << " ---" << std::endl;

        std::filesystem::path binPath(entry);
        deserializeWeights(cweights, bweights, w, h, entry);
        confusion.clear();
        confusion.assign(outSize, std::vector<int>(outSize, 0));

        for(size_t i = 0; i < totalInputs; ++i) {
            const auto& filePath = filePaths[i];
            // Prepare input
            std::vector<std::vector<float>> input = image2matrix(filePath.string(), isRGB);
            for(size_t r = 0; r < input.size(); r++) {
                for(size_t c = 0; c < input[r].size(); c++) {
                    input[r][c] /= 255.0f;
                }
            }
            std::string filename = filePath.stem().string();
            int label = std::stoi(filename.substr(filename.find_last_of('_') + 1));
            std::vector<float> target(outSize, 0.0f);
            if (label < outSize) {
                target[label] = 1.0f;
            }

            // Perform forward propagation
            #ifdef USE_CPU
                forprop(input);
            #elif USE_CU
                cuForprop(input);
            #elif USE_CL
                clForprop(input);
            #endif

            // Check prediction
            if(maxIndex(this->output) == static_cast<size_t>(label)) {
                correctPredictions++;
            }
            confusion[label][maxIndex(this->output)] += 1;
            // accumulate loss
            accLoss += crossEntropy(this->output, target);
            getScore(output, target, allScores.totalSumOfSquares, allScores.totalSumOfRegression, allScores.totalSumOfError);

            /*
            if((i + 1) % 1000 == 0 || (i + 1) == totalInputs) {
                float currentAccuracy = (float)correctPredictions / (i + 1);
                std::cout << "Processed " << i + 1 << "/" << totalInputs
                        << " \t Correct Prediction Percentage: " << currentAccuracy * 100.0f << "%\t"
                        << " | Avg Loss: " << accLoss / (i + 1.0f) << std::endl;
            }
            */
        }

        testPrg.testError = (totalInputs > 0) ? (accLoss / totalInputs) : 0.0f;
        testPrg.correctPredictions = correctPredictions;
        testPrg.testAccuracy = static_cast<float>(correctPredictions * 100) / totalInputs;
        testPrg.totalTestFiles = totalInputs;
        // extract name of entry .bin file and add it to following
        std::string pathForPrg = path2all + "/" + binPath.stem().string();
        logTestProgressToCSV(testPrg, path2test_progress);
        // evaluation
        allScores.sse = allScores.totalSumOfError / totalInputs;
        allScores.ssr = allScores.totalSumOfRegression / totalInputs;
        allScores.sst = allScores.totalSumOfSquares / totalInputs;
        allScores.r2 = allScores.ssr / allScores.sst;
        confData = {};
        confData = confusionMatrixFunc(confusion);
        epochDataToCsv(pathForPrg, confusion, confData, allScores, testPrg, true);

        synopsisFile << binPath.filename().string() << "," << correctPredictions << "," << testPrg.testError << "," << testPrg.testAccuracy << "\n";

        std::cout << "------ Result ------" << std::endl;
        std::cout << "Total Inputs: " << totalInputs << std::endl;
        std::cout << "Final Accuracy: " << testPrg.testAccuracy << "%" << std::endl;
        std::cout << "Final Average Loss: " << testPrg.testError << std::endl;
        std::cout << "Correct Predictions: " << correctPredictions << std::endl;
    }
    synopsisFile.close();
    std::cout << "Synopsis of all weights test saved to " << synopsisFilePath << std::endl;
    std::cout << "--- Test For All Weights Finished (mnn2d) ---" << std::endl;
}