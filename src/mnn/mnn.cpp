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
#ifdef USE_OPENCL
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
#ifdef USE_OPENCL
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
#ifdef USE_OPENCL
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

#ifdef _MSC_VER
   fopen_s(&this->binFile, fileAddress.c_str(), "rb+");
#else
   this->binFile = fopen(fileAddress.c_str(), "rb+");
#endif

   if (this->binFile) { // File exists
       fseek(this->binFile, 0, SEEK_END);
       long fileSize = ftell(this->binFile);
       rewind(this->binFile);

       long expectedFileSize = (long)(this->param * sizeof(float));

       if (fileSize == expectedFileSize) {
           // Parameters match, read weights and biases
           for (int i = 0; i < layers; ++i) {
               for (auto& row : cweights[i]) {
                   size_t read_count = fread(row.data(), sizeof(float), row.size(), this->binFile);
                   if (read_count != row.size()) {
                       std::cerr << "Error reading cweights[" << i << "]" << std::endl;
                       // Handle error (e.g., close file, throw exception)
                       fclose(this->binFile);
                       throw std::runtime_error("Error reading weights from file.");
                   }
               }
               for (auto& row : bweights[i]) {
                   size_t read_count = fread(row.data(), sizeof(float), row.size(), this->binFile);
                   if (read_count != row.size()) {
                       std::cerr << "Error reading bweights[" << i << "]" << std::endl;
                       // Handle error (e.g., close file, throw exception)
                       fclose(this->binFile);
                       throw std::runtime_error("Error reading weights from file.");
                   }
               }
           }
           std::cout << "Loaded weights from existing file." << std::endl;
       } else {
           std::cout << "File size mismatch. Expected " << expectedFileSize << ", found " << fileSize << ". Re-creating the file." << std::endl;
           fclose(this->binFile);
#ifdef _MSC_VER
           fopen_s(&this->binFile, fileAddress.c_str(), "wb+");
#else
           this->binFile = fopen(fileAddress.c_str(), "wb+");
#endif

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
    }
    else {
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