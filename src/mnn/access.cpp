#include <filesystem>
#include <fstream>
#include <stdexcept>
#include "mnn.hpp"

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