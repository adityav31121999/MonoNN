#include "mnn2d.hpp"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <filesystem>

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
    batchSize(1), binFileAddress(binFileAddress), epochs(100), iterations(0), learningRate(0.01f)
{
    this->trainPrg = {};
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
    // dimension = inh * width[0]
    cweights[0].resize(inw, std::vector<float>(width[0], 0.0f));
    bweights[0].resize(inw, std::vector<float>(width[0], 0.0f));
    cgradients[0].resize(inw, std::vector<float>(width[0], 0.0f));
    bgradients[0].resize(inw, std::vector<float>(width[0], 0.0f));
    for (int i = 1; i < layers-1; i++) {
        // dimension = width[i-1] * width[i]
        cweights[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
        bweights[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
        cgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
        bgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
    }
    // dimension = width[i] * outw
    cweights[layers-1].resize(width[layers-1], std::vector<float>(outw, 0.0f));
    bweights[layers-1].resize(width[layers-1], std::vector<float>(outw, 0.0f));
    cgradients[layers-1].resize(width[layers-1], std::vector<float>(outw, 0.0f));
    bgradients[layers-1].resize(width[layers-1], std::vector<float>(outw, 0.0f));

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
    // Initialize OpenCL context and command queue
    try {    
        // --- Enhanced OpenCL Initialization with Debug Info ---
        cl_int err;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found. Check your OpenCL installation and drivers.");
        }

        // Don't print details again if already done by another constructor
        // std::cout << "--- Found OpenCL Platforms (mnn2d) ---" << std::endl;
        // for(const auto& p : platforms) {
        //     std::cout << "Platform: " << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
        // }
        // std::cout << "-------------------------------------" << std::endl;

        clContext = cl::Context(CL_DEVICE_TYPE_DEFAULT, nullptr, nullptr, nullptr, &err); CL_CHECK(err);
        auto devices = clContext.getInfo<CL_CONTEXT_DEVICES>();
        if (devices.empty()) {
            throw std::runtime_error("No OpenCL devices found in the default context.");
        }
        clCommandQueue = cl::CommandQueue(clContext, devices[0], cl::QueueProperties::None, &err); CL_CHECK(err);
        createKernelsFromFile(clContext, kernelFiles, kernels);
        std::cout << "OpenCL kernels created successfully for mnn2d." << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cerr << "\n!! FATAL OPENCL INITIALIZATION ERROR (mnn2d) !!" << std::endl;
        std::cerr << e.what() << std::endl;
        throw;
    }
#elif USE_CU
    try {
        // --- Enhanced CUDA Initialization with Debug Info ---
        // print available devices
        int deviceCount = 0;
        cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("Failed to get CUDA device count: " + std::string(cudaGetErrorString(cudaStatus)));
        }
        if (deviceCount == 0) {
            throw std::runtime_error("No CUDA devices found. Check your CUDA installation and drivers.");
        }
        std::cout << "--- Found CUDA Devices (mnn2d) ---" << std::endl;
        for (int device = 0; device < deviceCount; ++device) {
            cudaDeviceProp deviceProp;
            cudaStatus = cudaGetDeviceProperties(&deviceProp, device);
            if (cudaStatus != cudaSuccess) {
                throw std::runtime_error("Failed to get properties for CUDA device " + std::to_string(device) + ": " + std::string(cudaGetErrorString(cudaStatus)));
            }
            std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
            std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
            std::cout << "  Max Threads Dim (1D, 2D, 3D): (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
            std::cout << "  Max Grid Size: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
        }
        std::cout << "----------------------------------" << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cerr << "\n!! FATAL CUDA INITIALIZATION ERROR (mnn2d) !!" << std::endl;
        std::cerr << e.what() << std::endl;
        throw;
    }
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
    batchSize(1), binFileAddress(binFileAddress), epochs(100), iterations(0), learningRate(0.01f)
{
    this->trainPrg = {};
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
    // dimension = inh * width[0]
    cweights[0].resize(inw, std::vector<float>(width[0], 0.0f));
    bweights[0].resize(inw, std::vector<float>(width[0], 0.0f));
    cgradients[0].resize(inw, std::vector<float>(width[0], 0.0f));
    bgradients[0].resize(inw, std::vector<float>(width[0], 0.0f));
    for (int i = 1; i < layers-1; i++) {
        // dimension = width[i-1] * width[i]
        cweights[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
        bweights[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
        cgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
        bgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
    }
    // dimension = width[i] * outw
    cweights[layers-1].resize(width[layers-1], std::vector<float>(outw, 0.0f));
    bweights[layers-1].resize(width[layers-1], std::vector<float>(outw, 0.0f));
    cgradients[layers-1].resize(width[layers-1], std::vector<float>(outw, 0.0f));
    bgradients[layers-1].resize(width[layers-1], std::vector<float>(outw, 0.0f));

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
    // Initialize OpenCL context and command queue
    try {
        // --- Enhanced OpenCL Initialization with Debug Info ---
        cl_int err;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found. Check your OpenCL installation and drivers.");
        }

        // Don't print details again if already done by another constructor
        // std::cout << "--- Found OpenCL Platforms (mnn2d) ---" << std::endl;
        // for(const auto& p : platforms) {
        //     std::cout << "Platform: " << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
        // }
        // std::cout << "-------------------------------------" << std::endl;

        clContext = cl::Context(CL_DEVICE_TYPE_DEFAULT, nullptr, nullptr, nullptr, &err); CL_CHECK(err);
        auto devices = clContext.getInfo<CL_CONTEXT_DEVICES>();
        if (devices.empty()) {
            throw std::runtime_error("No OpenCL devices found in the default context.");
        }
        clCommandQueue = cl::CommandQueue(clContext, devices[0], cl::QueueProperties::None, &err); CL_CHECK(err);
        createKernelsFromFile(clContext, kernelFiles, kernels);
        std::cout << "OpenCL kernels created successfully for mnn2d." << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cerr << "\n!! FATAL OPENCL INITIALIZATION ERROR (mnn2d) !!" << std::endl;
        std::cerr << e.what() << std::endl;
        throw;
    }
#elif USE_CU
    try {
        // --- Enhanced CUDA Initialization with Debug Info ---
        // print available devices
        int deviceCount = 0;
        cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("Failed to get CUDA device count: " + std::string(cudaGetErrorString(cudaStatus)));
        }
        if (deviceCount == 0) {
            throw std::runtime_error("No CUDA devices found. Check your CUDA installation and drivers.");
        }
        for (int device = 0; device < deviceCount; ++device) {
            cudaDeviceProp deviceProp;
            cudaStatus = cudaGetDeviceProperties(&deviceProp, device);
            if (cudaStatus != cudaSuccess) {
                throw std::runtime_error("Failed to get properties for CUDA device " + std::to_string(device) + ": " + std::string(cudaGetErrorString(cudaStatus)));
            }
        }
    }
    catch (const std::runtime_error& e) {
        std::cerr << "\n!! FATAL CUDA INITIALIZATION ERROR !!" << std::endl;
        std::cerr << e.what() << std::endl;
        throw; 
    }
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
    width(width), batchSize(1), binFileAddress(binFileAddress), epochs(100), iterations(0), learningRate(0.01f)
{
    this->trainPrg = {};
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
    // dimension = inw * width[0]
    cweights[0].resize(inw, std::vector<float>(width[0], 0.0f));
    bweights[0].resize(inw, std::vector<float>(width[0], 0.0f));
    cgradients[0].resize(inw, std::vector<float>(width[0], 0.0f));
    bgradients[0].resize(inw, std::vector<float>(width[0], 0.0f));
    for (int i = 1; i < layers-1; i++) {
        // dimension = width[i-1] * width[i]
        cweights[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
        bweights[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
        cgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
        bgradients[i].resize(width[i-1], std::vector<float>(width[i], 0.0f));
    }
    // dimension = width[i] * outw
    cweights[layers-1].resize(width[layers-2], std::vector<float>(outw, 0.0f));
    bweights[layers-1].resize(width[layers-2], std::vector<float>(outw, 0.0f));
    cgradients[layers-1].resize(width[layers-2], std::vector<float>(outw, 0.0f));
    bgradients[layers-1].resize(width[layers-2], std::vector<float>(outw, 0.0f));

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
    // Initialize OpenCL context and command queue
    try {
        // --- Enhanced OpenCL Initialization with Debug Info ---
        cl_int err;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found. Check your OpenCL installation and drivers.");
        }

        std::cout << "--- Found OpenCL Platforms (mnn2d) ---" << std::endl;
        for(const auto& p : platforms) {
            std::cout << "Platform: " << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
            std::vector<cl::Device> devices;
            p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            for(const auto& d : devices) {
                std::cout << "  - Device: " << d.getInfo<CL_DEVICE_NAME>() << std::endl;
            }
        }
        std::cout << "-------------------------------------" << std::endl;

        clContext = cl::Context(CL_DEVICE_TYPE_DEFAULT, nullptr, nullptr, nullptr, &err); CL_CHECK(err);
        auto devices = clContext.getInfo<CL_CONTEXT_DEVICES>();
        if (devices.empty()) {
            throw std::runtime_error("No OpenCL devices found in the default context.");
        }
        clCommandQueue = cl::CommandQueue(clContext, devices[0], cl::QueueProperties::None, &err); CL_CHECK(err);
        createKernelsFromFile(clContext, kernelFiles, kernels);
        std::cout << "OpenCL kernels created successfully for mnn2d." << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cerr << "\n!! FATAL OPENCL INITIALIZATION ERROR (mnn2d) !!" << std::endl;
        std::cerr << e.what() << std::endl;
        throw;
    }
#elif USE_CU
    try {
        // --- Enhanced CUDA Initialization with Debug Info ---
        // print available devices
        int deviceCount = 0;
        cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("Failed to get CUDA device count: " + std::string(cudaGetErrorString(cudaStatus)));
        }
        if (deviceCount == 0) {
            throw std::runtime_error("No CUDA devices found. Check your CUDA installation and drivers.");
        }
        std::cout << "--- Found CUDA Devices (mnn2d) ---" << std::endl;
        for (int device = 0; device < deviceCount; ++device) {
            cudaDeviceProp deviceProp;
            cudaStatus = cudaGetDeviceProperties(&deviceProp, device);
            if (cudaStatus != cudaSuccess) {
                throw std::runtime_error("Failed to get properties for CUDA device " + std::to_string(device) + ": " + std::string(cudaGetErrorString(cudaStatus)));
            }
            std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
            std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
            std::cout << "  Max Threads Dim (1D, 2D, 3D): (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
            std::cout << "  Max Grid Size: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
        }
        std::cout << "----------------------------------" << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cerr << "\n!! FATAL CUDA INITIALIZATION ERROR (mnn2d) !!" << std::endl;
        std::cerr << e.what() << std::endl;
        throw;
    }
#endif
}

/**
 * @brief Create or load binary file for weights and biases for mnn2d.
 * @param fileAddress Address of the binary file.
 */
void mnn2d::makeBinFile(const std::string &fileAddress)
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
	::makeBinFile(fileAddress, this->param);
	std::cout << "Binary file created successfully." << std::endl;
}