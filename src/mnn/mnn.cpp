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
 * @param datasetpath path to dataset
 */
mnn::mnn(int insize, int outsize, int layers, float order, std::string datasetpath) :
    order(order), inSize(insize), outSize(outsize), layers(layers), input(insize, 0.0f), 
    output(outsize, 0.0f), target(outsize, 0.0f), batchSize(1),
    epochs(100), iterations(0), learningRate(0.01f),
    binFileAddress(datasetpath + "/mnn1d/trainedWeigts.bin"),
    initialValues(datasetpath + "/mnn1d/initialisedWeights.bin")
{
    trainPrg = {};
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
    path2test_progress = datasetpath + "/mnn1d/testProgress.csv";
    path2progress = datasetpath + "/mnn1d/trainProgress.csv";
    makeBinFile(initialValues);
    makeBinFile(binFileAddress);
    std::cout << "Network initialized with " << param << " parameters." 
              << " Total Size: " << sizeof(float) * param / (1024.0 * 1024.0) << " MB"<< std::endl;
#ifdef USE_CL
    try {
        // --- Enhanced OpenCL Initialization with Debug Info ---
        cl_int err;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found. Check your OpenCL installation and drivers.");
        }

        std::cout << "--- Found OpenCL Platforms ---" << std::endl;
        for(const auto& p : platforms) {
            std::cout << "Platform: " << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
            std::vector<cl::Device> devices;
            p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            for(const auto& d : devices) {
                std::cout << "  - Device: " << d.getInfo<CL_DEVICE_NAME>() << std::endl;
            }
        }
        std::cout << "----------------------------" << std::endl;

        // Attempt to create a context on the default device
        clContext = cl::Context(CL_DEVICE_TYPE_DEFAULT, nullptr, nullptr, nullptr, &err); CL_CHECK(err);
        auto devices = clContext.getInfo<CL_CONTEXT_DEVICES>();
        if (devices.empty()) {
            throw std::runtime_error("No OpenCL devices found in the default context.");
        }
        clCommandQueue = cl::CommandQueue(clContext, devices[0], 0, &err); CL_CHECK(err);
        createKernelsFromFile(clContext, kernelFiles, kernels);
        std::cout << "OpenCL kernels created successfully." << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cerr << "\n!! FATAL OPENCL INITIALIZATION ERROR !!" << std::endl;
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
 * @brief Constructor for the mnn class in-out and layer size.
 * @param insize Input size.
 * @param outsize Output size.
 * @param dim dimension of hidden layers
 * @param layers Number of hidden layers.
 * @param order order of monomial
 */
mnn::mnn(int insize, int outsize, int dim, int layers, float order, std::string datasetpath) :
    order(order), inSize(insize), outSize(outsize), layers(layers), input(insize, 0.0f), 
    output(outsize, 0.0f), target(outsize, 0.0f), batchSize(1),
    epochs(100), iterations(0), learningRate(0.01f),
    binFileAddress(datasetpath + "/mnn1d/trainedWeigts.bin"),
    initialValues(datasetpath + "/mnn1d/initialisedWeights.bin")
{
    trainPrg = {};
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
    path2test_progress = datasetpath + "/mnn1d/testProgress.csv";
    path2progress = datasetpath + "/mnn1d/trainProgress.csv";
    makeBinFile(initialValues);
    makeBinFile(binFileAddress);
    std::cout << "Network initialized with " << param << " parameters." 
              << " Total Size: " << sizeof(float) * param / (1024.0 * 1024.0) << " MB"<< std::endl;
#ifdef USE_CL
    try {
        // --- Enhanced OpenCL Initialization with Debug Info ---
        cl_int err;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found. Check your OpenCL installation and drivers.");
        }

        // Don't print details again if already done by another constructor
        // std::cout << "--- Found OpenCL Platforms ---" << std::endl;
        // for(const auto& p : platforms) {
        //     std::cout << "Platform: " << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
        // }
        // std::cout << "----------------------------" << std::endl;

        // Attempt to create a context on the default device
        clContext = cl::Context(CL_DEVICE_TYPE_DEFAULT, nullptr, nullptr, nullptr, &err); CL_CHECK(err);

        auto devices = clContext.getInfo<CL_CONTEXT_DEVICES>();
        if (devices.empty()) {
           throw std::runtime_error("No OpenCL devices found in the default context.");
        }
        clCommandQueue = cl::CommandQueue(clContext, devices[0], 0, &err); CL_CHECK(err);

        createKernelsFromFile(clContext, kernelFiles, kernels);
        std::cout << "OpenCL kernels created successfully." << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cerr << "\n!! FATAL OPENCL INITIALIZATION ERROR !!" << std::endl;
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
 * @brief Constructor for the mnn class with layer specifications.
 * @param insize Input size.
 * @param outsize Output size.
 * @param dim dimension of hidden layers
 * @param layers Number of hidden layers.
 * @param order order of monomial
 */
mnn::mnn(int insize, int outsize, std::vector<int> width, float order, std::string datasetpath) : 
    order(order), inSize(insize), outSize(outsize), width(width), layers(width.size()),
    input(insize, 0.0f), output(outsize, 0.0f), target(outsize, 0.0f), batchSize(1),
    epochs(100), iterations(0), learningRate(0.01f),
    binFileAddress(datasetpath + "/mnn1d/trainedWeigts.bin"),
    initialValues(datasetpath + "/mnn1d/initialisedWeights.bin")
{
    trainPrg = {};
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
    path2test_progress = datasetpath + "/mnn1d/testProgress.csv";
    path2progress = datasetpath + "/mnn1d/trainProgress.csv";
    path2SessionDir = datasetpath + "/mnn1d/session";
    path2EpochDir = datasetpath + "/mnn1d/epoch";
    path2PreDir = datasetpath + "/mnn1d/pre";
    makeBinFile(initialValues);
    makeBinFile(binFileAddress);
    std::cout << "Network initialized with " << param << " parameters." 
              << " Total Size: " << sizeof(float) * param / (1024.0 * 1024.0) << " MB"<< std::endl;
#ifdef USE_CL
    try {
        // --- Enhanced OpenCL Initialization with Debug Info ---
        cl_int err;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found. Check your OpenCL installation and drivers.");
        }

        // Attempt to create a context on the default device
        clContext = cl::Context(CL_DEVICE_TYPE_DEFAULT, nullptr, nullptr, nullptr, &err); CL_CHECK(err);
 
        // Get devices and create a command queue
        auto devices = clContext.getInfo<CL_CONTEXT_DEVICES>();
        if (devices.empty()) {
            throw std::runtime_error("No OpenCL devices found in the default context.");
        }
        clCommandQueue = cl::CommandQueue(clContext, devices[0], 0, &err); CL_CHECK(err);

        createKernelsFromFile(clContext, kernelFiles, kernels);
        std::cout << "OpenCL kernels created successfully." << std::endl;
 
    } catch (const std::runtime_error& e) {
        std::cerr << "\n!! FATAL OPENCL INITIALIZATION ERROR !!" << std::endl;
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
 * @brief Create or load or resize binary file for weights and biases.
 * @param fileAddress Address of the binary file.
 */
void mnn::makeBinFile(const std::string &fileAddress)
{
	long expectedFileSize = (long)(this->param * sizeof(float));

    // Ensure the directory exists before trying to access the file.
    std::filesystem::path filePath(fileAddress);
    std::filesystem::path dirPath = filePath.parent_path();
    if (!dirPath.empty() && !std::filesystem::exists(dirPath)) {
        std::cout << "Directory " << dirPath << " does not exist. Creating it." << std::endl;
        if (!std::filesystem::create_directories(dirPath)) {
            throw std::runtime_error("MNN CONSTRUCTOR: could not create directory: " + dirPath.string());
        }
    }
    else {
        std::cout << "Directory " << dirPath << " exists." << std::endl;
    }

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
				std::cerr << "MNN CONSTRUCTOR: Error loading existing weights file: " << e.what() << ". Re-creating file." << std::endl;
			}
		}
		else {
			// If it exists but size is wrong, log it and proceed to create a new one.
            if(expectedFileSize != fileSize) {
                std::cout << "File size mismatch. Expected " << expectedFileSize << " bytes, found " 
                          << fileSize << ". Re-creating file." << std::endl;
            }
            // If the file does not exist, is the wrong size, or failed to load, create a new one.
            std::cout << "Re-creating weights file: " << fileAddress << std::endl;
            FILE* file = nullptr;
            #if defined(_WIN32) || defined(_MSC_VER)
                fopen_s(&file, fileAddress.c_str(), "wb");
            #else
                file = fopen(fileAddress.c_str(), "wb");
            #endif

            if (!file) {
                throw std::runtime_error("MNN CONSTRUCTOR: Could not open file for writing: " + fileAddress);
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
            std::cout << "Binary file created successfully." << std::endl;
            std::cout << "Provide weight initialisation type: ";
            std::cin >> weightUpdateType;
            std::cout << std::endl;
            initiateWeights(weightUpdateType);
            serializeWeights(cweights, bweights, fileAddress);
		}
	}
    else {
        // If it doesn't exist, create a new
        std::cout << "Creating new weights file: " << fileAddress << std::endl;
        FILE* file = nullptr;
        #if defined(_WIN32) || defined(_MSC_VER)
            fopen_s(&file, fileAddress.c_str(), "wb");
        #else
            file = fopen(fileAddress.c_str(), "wb");
        #endif

        if (!file) {
            throw std::runtime_error("MNN CONSTRUCTOR: Could not open file for writing: " + fileAddress);
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
        std::cout << "Binary file created successfully." << std::endl;
        std::cout << "Provide weight initialisation type: ";
        std::cin >> weightUpdateType;
        std::cout << std::endl;
        initiateWeights(weightUpdateType);
        serializeWeights(cweights, bweights, fileAddress);
    }
    saveNetwork();
}