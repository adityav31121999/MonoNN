#include "mnn.hpp"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <filesystem>

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
    alpha = 1.0f;
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
        cweights[i].resize(width[i-1], std::vector<float>(width[i]));
        bweights[i].resize(width[i-1], std::vector<float>(width[i]));
        cgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
        bgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
    }
    param = 0;
    for(int i = 0; i < layers; i++) {
        // c-weights
        param += static_cast<unsigned long long>(cweights[i].size() * cweights[i][0].size());
    }
    param *= 2; // b-weights
    makeBinFile(binFileAddress);
    std::cout << "Network initialized with " << param << " parameters." 
              << " Total Size: " << sizeof(float) * param / (1024.0 * 1024.0) << " MB"<< std::endl;
#ifdef USE_CL
    try {
        cl_int err;
        clContext = cl::Context(CL_DEVICE_TYPE_DEFAULT, nullptr, nullptr, nullptr, &err); CL_CHECK(err);
        auto devices = clContext.getInfo<CL_CONTEXT_DEVICES>();
        clCommandQueue = cl::CommandQueue(clContext, devices[0], 0, &err); CL_CHECK(err);
        createKernelsFromFile(clContext, "D:\\monoNN\\src\\mnn\\cl\\kernel.cl", kernels);
        std::cout << "OpenCL kernels created successfully." << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cerr << "\n!! FATAL OPENCL INITIALIZATION ERROR !!" << std::endl;
        std::cerr << e.what() << std::endl;
        throw; 
    }
#endif
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
    alpha = 1.0f;
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
        cweights[i].resize(width[i-1], std::vector<float>(width[i]));
        bweights[i].resize(width[i-1], std::vector<float>(width[i]));
        cgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
        bgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
    }
    param = 0;
    for(int i = 0; i < layers; i++) {
        // c-weights
        param += static_cast<unsigned long long>(cweights[i].size() * cweights[i][0].size());
    }
    param *= 2; // b-weights
    makeBinFile(binFileAddress);
    std::cout << "Network initialized with " << param << " parameters." 
              << " Total Size: " << sizeof(float) * param / (1024.0 * 1024.0) << " MB"<< std::endl;
#ifdef USE_CL
    try {
        // Initialize OpenCL context
        cl_int err;
        clContext = cl::Context(CL_DEVICE_TYPE_DEFAULT, nullptr, nullptr, nullptr, &err); CL_CHECK(err);
 
        auto devices = clContext.getInfo<CL_CONTEXT_DEVICES>();
        clCommandQueue = cl::CommandQueue(clContext, devices[0], 0, &err); CL_CHECK(err);
 
        createKernelsFromFile(clContext, "D:\\monoNN\\src\\mnn\\cl\\kernel.cl", kernels);
        std::cout << "OpenCL kernels created successfully." << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cerr << "\n!! FATAL OPENCL INITIALIZATION ERROR !!" << std::endl;
        std::cerr << e.what() << std::endl;
        throw; 
    }
#endif
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
    alpha = 1.0f;
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
        cweights[i].resize(width[i-1], std::vector<float>(width[i]));
        bweights[i].resize(width[i-1], std::vector<float>(width[i]));
        cgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
        bgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
    }
    param = 0;
    for(int i = 0; i < layers; i++) {
        // c-weights
        param += static_cast<unsigned long long>(cweights[i].size() * cweights[i][0].size());
    }
    param *= 2; // b-weights
    makeBinFile(binFileAddress);
    std::cout << "Network initialized with " << param << " parameters." 
              << " Total Size: " << sizeof(float) * param / (1024.0 * 1024.0) << " MB"<< std::endl;
#ifdef USE_CL
    try {
        // Initialize OpenCL context
        cl_int err;
        clContext = cl::Context(CL_DEVICE_TYPE_DEFAULT, nullptr, nullptr, nullptr, &err); CL_CHECK(err);
 
        // Get devices and create a command queue
        auto devices = clContext.getInfo<CL_CONTEXT_DEVICES>();
        clCommandQueue = cl::CommandQueue(clContext, devices[0], 0, &err); CL_CHECK(err);
 
        // Now, call the function that can also throw an exception
        // IMPORTANT: Change the hardcoded path to a relative one if possible
        createKernelsFromFile(clContext, "D:\\monoNN\\src\\mnn\\cl\\kernel.cl", kernels);
        
        std::cout << "OpenCL kernels created successfully." << std::endl;
 
    } catch (const std::runtime_error& e) {
        std::cerr << "\n!! FATAL OPENCL INITIALIZATION ERROR !!" << std::endl;
        std::cerr << e.what() << std::endl;
        throw; 
    }
#endif
}

/**
 * @brief Create or load binary file for weights and biases.
 * @param fileAddress Address of the binary file.
 */
void mnn::makeBinFile(const std::string &fileAddress)
{
	this->binFileAddress = fileAddress;
	long expectedFileSize = (long)(this->param * sizeof(float));

	// Check if the file exists and has the correct size.
	if (std::filesystem::exists(fileAddress)) {
		long fileSize = std::filesystem::file_size(fileAddress);
		if (fileSize == expectedFileSize) {
			// If it exists and size is correct, load it.
			std::cout << "Loading weights from existing file: " << fileAddress << std::endl;
			try {
				loadNetwork();
				std::cout << "Binary file ready at: " << fileAddress << " & size: " << fileSize / (1024.0 * 1024.0) << " MB" << std::endl;
				return; // Success, we are done.
			}
			catch (const std::runtime_error& e) {
				std::cerr << "Error loading existing weights file: " << e.what() << ". Re-creating file." << std::endl;
			}
		}
		else {
			// If it exists but size is wrong, log it and proceed to create a new one.
			std::cout << "File size mismatch. Expected " << expectedFileSize << " bytes, found " << fileSize << ". Re-creating file." << std::endl;
		}
	}

	// If the file does not exist, is the wrong size, or failed to load, create a new one.
	std::cout << "Creating new weights file: " << fileAddress << std::endl;
	::makeBinFile(fileAddress, this->param); // Use the global makeBinFile function from access.cpp
	std::cout << "Binary file created successfully." << std::endl;
}