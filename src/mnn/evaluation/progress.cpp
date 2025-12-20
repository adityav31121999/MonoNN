#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "progress.hpp"

/**
 * @brief Reads the training stage configuration from a CSV file.
 * @param filepath Path to the traintest.csv file.
 * @return A vector of StageInfo structs.
 */
std::vector<StageInfo> readTrainTestCsv(const std::string& filepath) {
    std::vector<StageInfo> stages;
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) return stages;

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string segment;
        StageInfo info;
        
        if (!std::getline(ss, segment, ',')) continue;
        info.name = segment;
        
        if (!std::getline(ss, segment, ',')) continue;
        try { info.id = std::stoi(segment); } catch (...) { continue; }
        
        if (!std::getline(ss, segment, ',')) continue;
        try { info.status = std::stoi(segment); } catch (...) { continue; }
        
        stages.push_back(info);
    }
    return stages;
}

/**
 * @brief Writes the training stage configuration to a CSV file.
 * @param filepath Path to the traintest.csv file.
 * @param stages Vector of StageInfo structs to write.
 */
void writeTrainTestCsv(const std::string& filepath, const std::vector<StageInfo>& stages) {
    std::ofstream ofs(filepath);
    for (const auto& s : stages) {
        ofs << s.name << ", " << s.id << ", " << s.status << "\n";
    }
}

/**
 * @brief Appends the current training progress as a new row in a CSV file.
 *        If the file doesn't exist, it creates it and adds a header.
 * @param p The progress struct to log.
 * @param filePath The path to the output CSV file.
 * @return true if logging was successful, false otherwise.
 */
bool logProgressToCSV(const progress& p, const std::string& filePath) {
    // Check if the file is empty to decide whether to write the header
    bool fileIsEmpty = false;
    {
        std::ifstream file(filePath);
        // A new or empty file will have the "peek" character be EOF
        fileIsEmpty = (file.peek() == std::ifstream::traits_type::eof());
    }

    // Open the file in append mode (std::ios::app)
    std::ofstream file(filePath, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for appending: " << filePath << std::endl;
        return false;
    }

    // If the file was empty, write the header first
    if (fileIsEmpty) {
        file << "epoch,batchSize,sessionSize,totalTrainFiles,filesProcessed,"
             << "currentLearningRate,loss,accLoss,trainingPredictions,correctPredPercent,totalCycleCount,"
             << "sessionCount,timeForCurrentSession,timeTakenForTraining,\n";
    }

    // Append the data row
    file << p.epoch << "," << p.batchSize << "," << p.sessionSize << "," << p.totalTrainFiles << ","
         << p.filesProcessed << "," << p.currentLearningRate << "," << p.loss << "," << p.accLoss << ","
         << p.trainingPredictions << "," << p.correctPredPercent << ","  << p.totalCycleCount << ","
         << p.sessionCount << "," << p.timeForCurrentSession << "," << p.timeTakenForTraining
         << "," << "\n";

    file.close();
    return true;
}


/**
 * @brief Loads the most recent training progress from the last line of a CSV log file.
 * @param p The progress struct to populate with the loaded data.
 * @param filePath The path to the input CSV file.
 * @return true if loading was successful, false otherwise.
 */
bool loadLastProgress(progress& p, const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        // This is not necessarily an error if it's the first run
        std::cout << "Info: Progress file not found. Starting a new training log." << std::endl;
        return false;
    }

    std::string line;
    std::string lastLine;
    
    // Read the file line by line to get the very last one
    while (std::getline(file, line)) {
        // Basic check to skip empty lines
        if (!line.empty()) {
            lastLine = line;
        }
    }

    // Check if we actually read anything (the file might just have a header)
    if (lastLine.empty() || lastLine.find("sessionSize") != std::string::npos) { // Make sure last line isn't the header
        std::cerr << "Warning: Progress file contains no valid data rows to load." << std::endl;
        return false;
    }

    std::stringstream ss(lastLine);
    std::string token;
    
    try {
        // The order of reading must exactly match the order of writing
        std::getline(ss, token, ','); p.epoch = std::stoul(token);
        std::getline(ss, token, ','); p.batchSize = std::stoi(token);
        std::getline(ss, token, ','); p.sessionSize = std::stoul(token);
        std::getline(ss, token, ','); p.totalTrainFiles = std::stoul(token);
        std::getline(ss, token, ','); p.filesProcessed = std::stoul(token);
        std::getline(ss, token, ','); p.currentLearningRate = std::stof(token);
        std::getline(ss, token, ','); p.loss = std::stof(token);
        std::getline(ss, token, ','); p.accLoss = std::stod(token);
        std::getline(ss, token, ','); p.trainingPredictions = std::stoul(token);
        std::getline(ss, token, ','); p.correctPredPercent = std::stof(token);
        std::getline(ss, token, ','); p.totalCycleCount = std::stoull(token);
        std::getline(ss, token, ','); p.sessionCount = std::stoul(token);
        std::getline(ss, token, ','); p.timeForCurrentSession = std::stod(token);
        std::getline(ss, token, ','); p.timeTakenForTraining = std::stod(token);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: Failed to parse last progress line. " << e.what() << std::endl;
        return false;
    }

    file.close();
    return true;
}

/**
 * @brief Appends the current testing progress as a new row in a CSV file.
 * @param p The test_progress struct to log.
 * @param filePath The path to the output CSV file.
 * @return true if logging was successful, false otherwise.
 */
bool logTestProgressToCSV(const test_progress& p, const std::string& filePath) {
    bool fileIsEmpty = false;
    {
        std::ifstream file(filePath);
        fileIsEmpty = (file.peek() == std::ifstream::traits_type::eof());
    }

    std::ofstream file(filePath, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for appending: " << filePath << std::endl;
        return false;
    }

    if (fileIsEmpty) {
        file << "totalTestFiles,testError,testAccuracy,correctPredictions\n";
    }

    file << p.totalTestFiles << "," 
         << p.testError << ","
         << p.testAccuracy << "," 
         << p.correctPredictions << "\n";

    file.close();
    return true;
}

/**
 * @brief Loads the most recent testing progress from the last line of a CSV log file.
 * @param p The test_progress struct to populate.
 * @param filePath The path to the input CSV file.
 * @return true if loading was successful, false otherwise.
 */
bool loadLastTestProgress(test_progress& p, const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cout << "Info: Test progress file not found. Starting a new test log." << std::endl;
        return false;
    }

    std::string line;
    std::string lastLine;

    while (std::getline(file, line)) {
        if (!line.empty()) {
            lastLine = line;
        }
    }

    if (lastLine.empty() || lastLine.find("totalTestFiles") != std::string::npos) {
        std::cerr << "Warning: Test progress file contains no valid data rows to load." << std::endl;
        return false;
    }

    std::stringstream ss(lastLine);
    std::string token;

    try {
        std::getline(ss, token, ','); p.totalTestFiles = std::stoul(token);
        std::getline(ss, token, ','); p.testError = std::stof(token);
        std::getline(ss, token, ','); p.testAccuracy = std::stof(token);
        std::getline(ss, token, ','); p.correctPredictions = std::stoul(token);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: Failed to parse last test progress line. " << e.what() << std::endl;
        return false;
    }

    file.close();
    return true;
}
