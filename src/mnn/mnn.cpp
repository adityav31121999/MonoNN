#include "mnn.hpp"
#include <stdexcept>
#include <iostream>

/**
 * @brief Constructor for the mnn class with only in-out size.
 * @param insize Input size.
 * @param outsize Output size.
 * @param layers Number of hidden layers.
 * @param order order of monomial
 */
mnn::mnn(int insize, int outsize, int layers, float order, std::string binFileAddress) :
    order(order), inSize(insize), outSize(outsize), layers(layers), input(insize, 0.0f), 
    output(outsize, 0.0f), target(outsize, 0.0f), batchSize(1), binFileAddress(binFileAddress)
{
    // set width of hidden layers and dot products
    // int dim = (insize > outsize) ? insize : outsize;     // (optional)
    int dim = (insize + outsize) / 2;
    width.resize(layers, dim);
    width[layers - 1] = outsize;
    // initialize weights
    cweights.resize(layers);
    bweights.resize(layers);
    cgradients.resize(layers);
    bgradients.resize(layers);
    dotProds.resize(layers);
    activate.resize(layers);
    // dot product and their activations
    for (int i = 0; i < layers; i++) {
        dotProds[i].resize(width[i]);
        activate[i].resize(width[i]);
    }
    // first to last layers
    cweights[0].resize(insize, std::vector<float>(width[0]));
    bweights[0].resize(insize, std::vector<float>(width[0]));
    cgradients[0].resize(insize, std::vector<float>(width[0]));
    bgradients[0].resize(insize, std::vector<float>(width[0]));
    for (int i = 1; i < layers; i++) {
        // dimension = width[i-1] * width[i]
        cweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        bweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        cgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
        bgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
    }
    cweights[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    bweights[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    cgradients[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    bgradients[layers-1].resize(width[layers-1], std::vector<float>(outsize));

    param = 0;
    for(int i = 0; i < layers; i++) {
        // c-weights
        param += static_cast<unsigned long long>(cweights[i].size() * cweights[i][0].size());
    }
    param *= 2; // b-weights
    makeBinFile(binFileAddress);
    std::cout << "Network initialized with " << param << " parameters." 
              << " Total Size: " << sizeof(float) * param / (1024.0 * 1024.0) << " MB"<< std::endl;
}


/**
 * @brief Constructor for the mnn class in-out and layer size.
 * @param insize Input size.
 * @param outsize Output size.
 * @param dim dimension of hidden layers
 * @param layers Number of hidden layers.
 * @param order order of monomial
 */
mnn::mnn(int insize, int outsize, int dim, int layers, float order, std::string binFileAddress) :
    order(order), inSize(insize), outSize(outsize), layers(layers), input(insize, 0.0f), 
    output(outsize, 0.0f), target(outsize, 0.0f), batchSize(1), binFileAddress(binFileAddress)
{
    // set width of hidden layers and dot products
    width.resize(layers, dim);
    width[layers - 1] = outsize;
    // initialize weights
    cweights.resize(layers);
    bweights.resize(layers);
    cgradients.resize(layers);
    bgradients.resize(layers);
    dotProds.resize(layers);
    activate.resize(layers);
    // dot product and their activations
    for (int i = 0; i < layers; i++) {
        dotProds[i].resize(width[i]);
        activate[i].resize(width[i]);
    }
    // first to last layers
    cweights[0].resize(insize, std::vector<float>(width[0]));
    bweights[0].resize(insize, std::vector<float>(width[0]));
    cgradients[0].resize(insize, std::vector<float>(width[0]));
    bgradients[0].resize(insize, std::vector<float>(width[0]));
    for (int i = 1; i < layers-1; i++) {
        // dimension = width[i-1] * width[i]
        cweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        bweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        cgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
        bgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
    }
    cweights[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    bweights[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    cgradients[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    bgradients[layers-1].resize(width[layers-1], std::vector<float>(outsize));

    param = 0;
    for(int i = 0; i < layers; i++) {
        // c-weights
        param += static_cast<unsigned long long>(cweights[i].size() * cweights[i][0].size());
    }
    param *= 2; // b-weights
    makeBinFile(binFileAddress);
    std::cout << "Network initialized with " << param << " parameters." 
              << " Total Size: " << sizeof(float) * param / (1024.0 * 1024.0) << " MB"<< std::endl;
}


/**
 * @brief Constructor for the mnn class with layer specifications.
 * @param insize Input size.
 * @param outsize Output size.
 * @param dim dimension of hidden layers
 * @param layers Number of hidden layers.
 * @param order order of monomial
 */
mnn::mnn(int insize, int outsize, std::vector<int> width, float order, std::string binFileAddress) : 
    order(order), inSize(insize), outSize(outsize), width(width), layers(width.size()),
    input(insize, 0.0f), output(outsize, 0.0f), target(outsize, 0.0f), batchSize(1), binFileAddress(binFileAddress)
{
    // initialize weights
    cweights.resize(layers);
    bweights.resize(layers);
    cgradients.resize(layers);
    bgradients.resize(layers);
    dotProds.resize(layers);
    activate.resize(layers);
    // dot product and their activations
    for (int i = 0; i < layers; i++) {
        dotProds[i].resize(width[i]);
        activate[i].resize(width[i]);
    }
    // first to last layers
    cweights[0].resize(insize, std::vector<float>(width[0]));
    bweights[0].resize(insize, std::vector<float>(width[0]));
    cgradients[0].resize(insize, std::vector<float>(width[0]));
    bgradients[0].resize(insize, std::vector<float>(width[0]));
    for (int i = 1; i < layers-1; i++) {
        // dimension = width[i-1] * width[i]
        cweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        bweights[i].resize(width[i], std::vector<float>(width[i + 1]));
        cgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
        bgradients[i].resize(width[i], std::vector<float>(width[i + 1], 0.0f));
    }
    cweights[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    bweights[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    cgradients[layers-1].resize(width[layers-1], std::vector<float>(outsize));
    bgradients[layers-1].resize(width[layers-1], std::vector<float>(outsize));

    param = 0;
    for(int i = 0; i < layers; i++) {
        // c-weights
        param += static_cast<unsigned long long>(cweights[i].size() * cweights[i][0].size());
    }
    param *= 2; // b-weights
    makeBinFile(binFileAddress);
    std::cout << "Network initialized with " << param << " parameters." 
              << " Total Size: " << sizeof(float) * param / (1024.0 * 1024.0) << " MB"<< std::endl;
}

/**
 * @brief Create or load binary file for weights and biases.
 * @param fileAddress Address of the binary file.
 */
void mnn::makeBinFile(const std::string &fileAddress)
{
    this->binFileAddress = fileAddress;

#ifdef _MSC_VER
    fopen_s(&this->binFile, fileAddress.c_str(), "rb+");
#else
    this->binFile = fopen(fileAddress.c_str(), "rb+");
#endif

    if (this->binFile) { // File exists
        fseek(this->binFile, 0, SEEK_END);
        long fileSize = ftell(this->binFile);
        rewind(this->binFile);

        if (fileSize == (long)(this->param * sizeof(float))) {
            // Parameters match, read weights and biases
            for (int i = 0; i < layers; ++i) {
                for (auto& row : cweights[i]) {
                    fread(row.data(), sizeof(float), row.size(), this->binFile);
                }
                for (auto& row : bweights[i]) {
                    fread(row.data(), sizeof(float), row.size(), this->binFile);
                }
            }
        } else {
            // Size does not match, re-create the file
            fclose(this->binFile);
#ifdef _MSC_VER
            fopen_s(&this->binFile, fileAddress.c_str(), "wb+");
#else
            this->binFile = fopen(fileAddress.c_str(), "wb+");
#endif
            if (this->binFile) {
                for (int i = 0; i < layers; ++i) {
                    for (auto& row : cweights[i]) fwrite(row.data(), sizeof(float), row.size(), this->binFile);
                    for (auto& row : bweights[i]) fwrite(row.data(), sizeof(float), row.size(), this->binFile);
                }
            }
        }
    } else {
        // File does not exist, create it
#ifdef _MSC_VER
        fopen_s(&this->binFile, fileAddress.c_str(), "wb+");
#else
        this->binFile = fopen(fileAddress.c_str(), "wb+");
#endif
        if (this->binFile) {
            // You might want to initialize weights before writing them
            for (int i = 0; i < layers; ++i) {
                for (auto& row : cweights[i]) fwrite(row.data(), sizeof(float), row.size(), this->binFile);
                for (auto& row : bweights[i]) fwrite(row.data(), sizeof(float), row.size(), this->binFile);
            }
        }
    }

    if (!this->binFile) {
        throw std::runtime_error("Could not open or create binary file: " + fileAddress);
    }

    rewind(this->binFile); // Rewind for future operations
    fseek(this->binFile, 0, SEEK_END);
    long currentFileSize = ftell(this->binFile);
    rewind(this->binFile);
    std::cout << "Binary file ready at: " << fileAddress << " & size: " << currentFileSize / (1024.0 * 1024.0) << " MB" << std::endl;
}