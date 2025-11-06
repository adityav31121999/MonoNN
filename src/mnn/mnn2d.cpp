#include "mnn.hpp"
#include <stdexcept>
#include <iostream>

/**
 * @brief Constructor for the mnn2d class in-out size.
 * @param inw Input width.
 * @param inh Input height.
 * @param outw Output width.
 * @param outh Output height.
 * @param layers Number of hidden layers.
 * @param order Order of the monomial.
 */
mnn2d::mnn2d(int inw, int inh, int outw, int layers, float order, std::string binFileAddress) :
    order(order), inWidth(inw), inHeight(inh), outWidth(outw), layers(layers),
    batchSize(1), binFileAddress(binFileAddress)
{
    // set hidden layers width and height
    int dim = (inw + outw) / 2;
    width.resize(layers, dim);
    width[layers - 1] = outw;

    // initialize weights
    cweights.resize(layers);
    bweights.resize(layers);
    cgradients.resize(layers);
    bgradients.resize(layers);
    dotProds.resize(layers);
    activate.resize(layers);

    // dot product and their activations
    for (int i = 0; i < layers-1; i++) {
        dotProds[i].resize(inh, std::vector<float>(width[i], 0.0f));
        activate[i].resize(inh, std::vector<float>(width[i], 0.0f));
    }
    dotProds[layers-1].resize(inh, std::vector<float>(outw, 0.0f));
    activate[layers-1].resize(inh, std::vector<float>(outw, 0.0f));

    // c,b-weights
    cweights[0].resize(inh, std::vector<float>(width[0]));
    bweights[0].resize(inh, std::vector<float>(width[0]));
    cgradients[0].resize(inh, std::vector<float>(width[0]));
    bgradients[0].resize(inh, std::vector<float>(width[0]));
    for (int i = 1; i < layers-1; i++) {
        // dimension = width[i-1] * width[i]
        cweights[i].resize(width[i-1], std::vector<float>(width[i]));
        bweights[i].resize(width[i-1], std::vector<float>(width[i]));
        cgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
        bgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
    }
    cweights[layers-1].resize(width[layers-1], std::vector<float>(outw));
    bweights[layers-1].resize(width[layers-1], std::vector<float>(outw));
    cgradients[layers-1].resize(width[layers-1], std::vector<float>(outw));
    bgradients[layers-1].resize(width[layers-1], std::vector<float>(outw));

    param = 0;
    for(int i = 0; i < layers; i++) {
        // c-weights
        param += static_cast<unsigned long long>(cweights[i].size() * cweights[i][0].size());
    }
    param *= 2; // b-weights
    makeBinFile(binFileAddress);
    std::cout << "Network initialized with " << param << " parameters." 
              << " Total Size: " << sizeof(float) * param / (1024.0 * 1024.0) << " MB"<< std::endl;
    #ifdef USE_OPENCL
    // Initialize OpenCL context and command queue
        clContext = cl::Context(CL_DEVICE_TYPE_ALL, nullptr, nullptr, nullptr, &err);
        createKernelsFromFile(clContext, "D:\\monoNN\\src\\mnn\\cl\\kernel.cl", kernels);
    #endif
}

/**
 * @brief Constructor for the mnn2d class with in-out and layer size.
 * @param inw Input width.
 * @param inh Input height.
 * @param outw Output width.
 * @param outh Output height.
 * @param dim Dimension of hidden layers.
 * @param layers Number of hidden layers.
 * @param order Order of the monomial.
 */
mnn2d::mnn2d(int inw, int inh, int outw, int dim, int layers, float order, std::string binFileAddress) :
    order(order), inWidth(inw), inHeight(inh), outWidth(outw), layers(layers),
    batchSize(1), binFileAddress(binFileAddress)
{
    // set hidden layers width and height
    width.resize(layers, dim);
    width[layers - 1] = outw;

    // initialize weights
    cweights.resize(layers);
    bweights.resize(layers);
    cgradients.resize(layers);
    bgradients.resize(layers);
    dotProds.resize(layers);
    activate.resize(layers);

    // dot product and their activations
    for (int i = 0; i < layers-1; i++) {
        dotProds[i].resize(inh, std::vector<float>(width[i], 0.0f));
        activate[i].resize(inh, std::vector<float>(width[i], 0.0f));
    }
    dotProds[layers-1].resize(inh, std::vector<float>(outw, 0.0f));
    activate[layers-1].resize(inh, std::vector<float>(outw, 0.0f));

    // c,b-weights
    cweights[0].resize(inh, std::vector<float>(width[0]));
    bweights[0].resize(inh, std::vector<float>(width[0]));
    cgradients[0].resize(inh, std::vector<float>(width[0]));
    bgradients[0].resize(inh, std::vector<float>(width[0]));
    for (int i = 1; i < layers-1; i++) {
        // dimension = width[i-1] * width[i]
        cweights[i].resize(width[i-1], std::vector<float>(width[i]));
        bweights[i].resize(width[i-1], std::vector<float>(width[i]));
        cgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
        bgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
    }
    cweights[layers-1].resize(width[layers-1], std::vector<float>(outw));
    bweights[layers-1].resize(width[layers-1], std::vector<float>(outw));
    cgradients[layers-1].resize(width[layers-1], std::vector<float>(outw));
    bgradients[layers-1].resize(width[layers-1], std::vector<float>(outw));

    param = 0;
    for(int i = 0; i < layers; i++) {
        // c-weights
        param += static_cast<unsigned long long>(cweights[i].size() * cweights[i][0].size());
    }
    param *= 2; // b-weights
    makeBinFile(binFileAddress);
    std::cout << "Network initialized with " << param << " parameters." 
              << " Total Size: " << sizeof(float) * param / (1024.0 * 1024.0) << " MB"<< std::endl;
    #ifdef USE_OPENCL
    // Initialize OpenCL context and command queue
        clContext = cl::Context(CL_DEVICE_TYPE_ALL, nullptr, nullptr, nullptr, &err);
        createKernelsFromFile(clContext, "D:\\monoNN\\src\\mnn\\cl\\kernel.cl", kernels);
    #endif
}

/**
 * @brief Constructor for the mnn2d class with layer specifications.
 * @param inw Input width.
 * @param inh Input height.
 * @param outw Output width.
 * @param outh Output height.
 * @param dim Dimension of hidden layers.
 */
mnn2d::mnn2d(int inw, int inh, int outw, std::vector<int> width, float order, std::string binFileAddress) :
    order(order), inWidth(inw), inHeight(inh), outWidth(outw), layers(width.size()),
    width(width), batchSize(1), binFileAddress(binFileAddress)
{
    input.resize(inh, std::vector<float>(inw, 0.0f));
    target.resize(outw, 0.0f);
    output.resize(outw, 0.0f);
    // initialize weights
    cweights.resize(layers);
    bweights.resize(layers);
    cgradients.resize(layers);
    bgradients.resize(layers);
    dotProds.resize(layers);
    activate.resize(layers);

    // dot product and their activations
    for (int i = 0; i < layers-1; i++) {
        dotProds[i].resize(inh, std::vector<float>(width[i], 0.0f));
        activate[i].resize(inh, std::vector<float>(width[i], 0.0f));
    }
    dotProds[layers-1].resize(inh, std::vector<float>(outw, 0.0f));
    activate[layers-1].resize(inh, std::vector<float>(outw, 0.0f));

    // c,b-weights
    cweights[0].resize(inh, std::vector<float>(width[0]));
    bweights[0].resize(inh, std::vector<float>(width[0]));
    cgradients[0].resize(inh, std::vector<float>(width[0]));
    bgradients[0].resize(inh, std::vector<float>(width[0]));
    for (int i = 1; i < layers-1; i++) {
        // dimension = width[i-1] * width[i]
        cweights[i].resize(width[i-1], std::vector<float>(width[i]));
        bweights[i].resize(width[i-1], std::vector<float>(width[i]));
        cgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
        bgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
    }
    cweights[layers-1].resize(width[layers-1], std::vector<float>(outw));
    bweights[layers-1].resize(width[layers-1], std::vector<float>(outw));
    cgradients[layers-1].resize(width[layers-1], std::vector<float>(outw));
    bgradients[layers-1].resize(width[layers-1], std::vector<float>(outw));

    param = 0;
    for(int i = 0; i < layers; i++) {
        // c-weights
        param += static_cast<unsigned long long>(cweights[i].size() * cweights[i][0].size());
    }
    param *= 2; // b-weights
    makeBinFile(binFileAddress);
    std::cout << "Network initialized with " << param << " parameters." 
              << " Total Size: " << sizeof(float) * param / (1024.0 * 1024.0) << " MB"<< std::endl;
    #ifdef USE_OPENCL
    // Initialize OpenCL context and command queue
        clContext = cl::Context(CL_DEVICE_TYPE_ALL, nullptr, nullptr, nullptr, &err);
        createKernelsFromFile(clContext, "D:\\monoNN\\src\\mnn\\cl\\kernel.cl", kernels);
    #endif
}

/**
 * @brief Create or load binary file for weights and biases for mnn2d.
 * @param fileAddress Address of the binary file.
 */
void mnn2d::makeBinFile(const std::string &fileAddress)
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
            // File size matches, read weights and biases
            for (int i = 0; i < layers; ++i) {
                for (auto& row : cweights[i]) fread(row.data(), sizeof(float), row.size(), this->binFile);
                for (auto& row : bweights[i]) fread(row.data(), sizeof(float), row.size(), this->binFile);
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