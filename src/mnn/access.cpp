#include <filesystem>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include "mnn.hpp"

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
        file << "batchSize,sessionSize,totalTrainFiles,totalTestFiles,batchSize,currentLearningRate,"
             << "loss,accLoss,filesProcessed,ongoingCycleCount,totalCycleCount,"
             << "totalSessionsOfTraining,timeForCurrentSession,timeTakenForTraining,"
             << "testError,testAccuracy,correctPredictions\n";
    }

    // Append the data row
    file << p.batchSize << "," << p.sessionSize << "," << p.totalTrainFiles << "," << p.totalTestFiles << ","
         << p.batchSize << "," << p.currentLearningRate << "," << p.loss << "," << p.accLoss << "," 
         << p.filesProcessed << "," << p.ongoingCycleCount << "," << p.totalCycleCount << "," 
         << p.totalSessionsOfTraining << "," << p.timeForCurrentSession << "," << p.timeTakenForTraining << ","
         << p.testError << "," << p.testAccuracy << "," << p.correctPredictions << "\n";

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
        std::getline(ss, token, ','); p.batchSize = std::stoi(token);
        std::getline(ss, token, ','); p.sessionSize = std::stoul(token);
        std::getline(ss, token, ','); p.totalTrainFiles = std::stoul(token);
        std::getline(ss, token, ','); p.totalTestFiles = std::stoul(token);
        std::getline(ss, token, ','); p.batchSize = std::stoi(token);
        std::getline(ss, token, ','); p.currentLearningRate = std::stof(token);
        std::getline(ss, token, ','); p.loss = std::stof(token);
        std::getline(ss, token, ','); p.accLoss = std::stod(token);
        std::getline(ss, token, ','); p.filesProcessed = std::stoul(token);
        std::getline(ss, token, ','); p.ongoingCycleCount = std::stoull(token);
        std::getline(ss, token, ','); p.totalCycleCount = std::stoull(token);
        std::getline(ss, token, ','); p.totalSessionsOfTraining = std::stoul(token);
        std::getline(ss, token, ','); p.timeForCurrentSession = std::stod(token);
        std::getline(ss, token, ','); p.timeTakenForTraining = std::stod(token);
        std::getline(ss, token, ','); p.testError = std::stof(token);
        std::getline(ss, token, ','); p.testAccuracy = std::stof(token);
        std::getline(ss, token, ','); p.correctPredictions = std::stoul(token);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: Failed to parse last progress line. " << e.what() << std::endl;
        return false;
    }

    file.close();
    return true;
}

/**
 * @brief Create a binary file initialized with zeros for weights and biases.
 * @param fileAddress Address of the binary file.
 * @param param Total number of parameters (weights + biases).
 */
void makeBinFile(const std::string& fileAddress, unsigned long long param) 
{
    FILE* file = nullptr;
#ifdef _MSC_VER
    fopen_s(&file, fileAddress.c_str(), "wb");
#else
    file = fopen(fileAddress.c_str(), "wb");
#endif

    if (!file) {
        throw std::runtime_error("Could not open file for writing: " + fileAddress);
    }

    // Write in chunks to avoid allocating a potentially huge vector all at once.
    const size_t chunkSize = 1024 * 1024; // 1M floats (4MB)
    std::vector<float> zeros(chunkSize, 0.0f);
    unsigned long long remaining = param;

    while (remaining > 0) {
        size_t toWrite = (remaining < chunkSize) ? static_cast<size_t>(remaining) : chunkSize;
        fwrite(zeros.data(), sizeof(float), toWrite, file);
        remaining -= toWrite;
    }

    fclose(file);
}

/**
 * @brief Serialize weights and biases to a binary file.
 * @param cweights Coefficients of the network.
 * @param bweights Biases of the network.
 * @param fileAddress Address of the binary file.
 */
void serializeWeights(const std::vector<std::vector<std::vector<float>>>& cweights,
                        const std::vector<std::vector<std::vector<float>>>& bweights,
                        const std::string& fileAddress)
{
    FILE* file = nullptr;
    #ifdef _MSC_VER
        fopen_s(&file, fileAddress.c_str(), "wb");
    #else
        file = fopen(fileAddress.c_str(), "wb");
    #endif

    if (!file) {
        throw std::runtime_error("Could not open file for writing: " + fileAddress);
    }

    // Serialize cweights
    for (const auto& layer_weights : cweights) {
        for (const auto& row : layer_weights) {
            fwrite(row.data(), sizeof(float), row.size(), file);
        }
    }

    // Serialize bweights immediately after
    for (const auto& layer_weights : bweights) {
        for (const auto& row : layer_weights) {
            fwrite(row.data(), sizeof(float), row.size(), file);
        }
    }

    fclose(file);
}


/**
 * @brief Deserialize weights and biases from a binary file (flat vectors).
 * @param cweights Coefficients of the network to be filled.
 * @param bweights Biases of the network to be filled.
 * @param fileAddress Address of the binary file.
 */
void deserializeWeights(std::vector<float>& cweights, std::vector<float>& bweights,
                        const std::string& fileAddress)
{
    FILE* file = nullptr;
    #ifdef _MSC_VER
        fopen_s(&file, fileAddress.c_str(), "rb");
    #else
        file = fopen(fileAddress.c_str(), "rb");
    #endif

    if (!file) {
        throw std::runtime_error("Could not open file for reading: " + fileAddress);
    }

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    rewind(file);

    if (fileSize == 0) {
        fclose(file);
        return;
    }

    // Since both have same number of values, divide by 2
    long totalFloats = fileSize / sizeof(float);
    long halfCount = totalFloats / 2;

    // Read cweights (first half)
    cweights.resize(halfCount);
    fread(cweights.data(), sizeof(float), halfCount, file);

    // Read bweights (second half)
    bweights.resize(halfCount);
    fread(bweights.data(), sizeof(float), halfCount, file);

    fclose(file);
}


/**
 * @brief Deserialize weights and biases from a binary file (3D vectors).
 * @param cweights Coefficients of the network to be filled.
 * @param bweights Biases of the network to be filled.
 * @param width Widths of each layer.
 * @param height Heights of each layer.
 * @param fileAddress Address of the binary file.
 */
void deserializeWeights(std::vector<std::vector<std::vector<float>>>& cweights,
                        std::vector<std::vector<std::vector<float>>>& bweights,
                        const std::vector<int>& width, const std::vector<int>& height,
                        const std::string& fileAddress)
{
    FILE* file = nullptr;
    #ifdef _MSC_VER
        fopen_s(&file, fileAddress.c_str(), "rb");
    #else
        file = fopen(fileAddress.c_str(), "rb");
    #endif

    if (!file) {
        throw std::runtime_error("Could not open file for reading: " + fileAddress);
    }

    // Deserialize cweights
    for (size_t i = 0; i < cweights.size(); ++i) {
        cweights[i].assign(height[i], std::vector<float>(width[i]));
        for (auto& row : cweights[i]) {
            fread(row.data(), sizeof(float), row.size(), file);
        }
    }

    // Deserialize bweights (continues from current position)
    for (size_t i = 0; i < bweights.size(); ++i) {
        bweights[i].assign(height[i], std::vector<float>(width[i]));
        for (auto& row : bweights[i]) {
            fread(row.data(), sizeof(float), row.size(), file);
        }
    }

    fclose(file);
}

// load data of networ from binary file
void mnn::loadNetwork() {
    std::vector<float> c(param/2, 0.0f);
    std::vector<float> b(param/2, 0.0f);
    deserializeWeights(c, b, binFileAddress);
    for(int i = 0; i < cweights.size(); i++) {
        for(int j = 0; j < cweights[i].size(); j++) {
            for(int k = 0; k < cweights[i][j].size(); k++) {
                cweights[i][j][k] = c[i*cweights[i].size()*cweights[i][j].size() + j*cweights[i][j].size() + k];
                bweights[i][j][k] = b[i*bweights[i].size()*bweights[i][j].size() + j*bweights[i][j].size() + k];
            }
        }
    }
}

// save data of network to binary file
void mnn::saveNetwork() {
    serializeWeights(cweights, bweights, binFileAddress);
}
