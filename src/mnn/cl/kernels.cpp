#ifdef USE_CL
#include "operators.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <map>
#include <iostream>
#include <CL/opencl.hpp>
#include <filesystem>

// Get the absolute source directory path from the CMake-defined macro
const std::filesystem::path src_dir = MONO_NN_SRC_DIR;

const std::vector<std::string> kernelFiles = [] {
    return std::vector<std::string>{
        (src_dir / "src/mnn/cl/activations.cl").string(),
        (src_dir / "src/mnn/cl/operators.cl").string(),
        (src_dir / "src/mnn/cl/forprop.cl").string(),
    };
}();

// Define and initialize the global kernelNames variable here.
const std::vector<std::string> kernelNames = {
    // actvations and derivative
    "sigmoid",
    "sigmoidDer",
    "softmax_reduce",
    "softmax_normalize",
    "softmax",
    "softmaxDer_normalize",
    "softmaxDer",
    // maths
    "add",
    "subtract",
    "scaleByValue",
    "power",
    "dPower",
    "meanPool",
    "maxPool",
    "transpose",
    "vecxvec2vec",
    "vecxvec2mat",
    "vecxmat2vec",
    "matxmat2mat",
    "matxvec2vec",
    "hadamard",
    "hadamard2",
    "matrix_vector_average",
    // forward propagation kernels
    "kernelLayerForward1",
    "kernelLayerForward2",
    "kernelLayerForward3",
    "kernelLayerForward4",
    "kernelLayerForwardBatch1",
    "kernelLayerForwardBatch2",
    "kernelLayerForwardBatch3",
    "kernelLayerForwardBatch4",
    // weight update kernels
    "kernelUpdateWeights",
    "kernelUpdateWeightsWithL1",
    "kernelUpdateWeightsWithL2",
    "kernelUpdateWeightsElasticNet",
    "kernelUpdateWeightsWeightDecay",
    "kernelUpdateWeightsDropout",
};

/**
 * @brief Reads an OpenCL kernel file, compiles it, and creates all kernel objects.
 * This function also checks and prints the maximum work-item and work-group sizes
 * for the device.
 * @param context A valid, initialized cl::Context object.
 * @param filePaths The path to the .cl kernel files.
 * @param kernels A reference to a std::map to be populated. The map will be
 *                  cleared before being filled.
 * @throws std::runtime_error if any step of the process fails.
 */
void createKernelsFromFile(const cl::Context& context, const std::vector<std::string>& filePaths, std::map<std::string, cl::Kernel>& kernelMap) {
    kernelMap.clear();

    // Get the devices associated with the context
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    if (devices.empty()) {
        throw std::runtime_error("No devices found in the provided OpenCL context.");
    }

    // --- New Feature: Check Maximum Work-Item and Work-Group Sizes ---
    // We'll query the first device in the context. In a multi-device context,
    // you might want to iterate over all devices.
    const cl::Device& device = devices[0];

    // Get maximum work-item sizes for each dimension
    std::vector<size_t> maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

    // Get the maximum work-group size
    size_t maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    std::cout << "--- OpenCL Device Capabilities ---" << std::endl;
    std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    if (maxWorkItemSizes.size() >= 1) {
        std::cout << "Max Work-Item Size (1D): " << maxWorkItemSizes[0] << std::endl;
    }
    if (maxWorkItemSizes.size() >= 2) {
        std::cout << "Max Work-Item Size (2D): (" << maxWorkItemSizes[0] << ", " << maxWorkItemSizes[1] << ")" << std::endl;
    }
    if (maxWorkItemSizes.size() >= 3) {
        std::cout << "Max Work-Item Size (3D): (" << maxWorkItemSizes[0] << ", " << maxWorkItemSizes[1] << ", " << maxWorkItemSizes[2] << ")" << std::endl;
    }
    std::cout << "Max Work-Group Size: " << maxWorkGroupSize << std::endl;
    std::cout << "----------------------------------" << std::endl;
    // --- End of New Feature ---

    cl::Program::Sources sources;
    std::vector<std::string> sourceCodes; // To keep the string data alive

    // --- Step 1: Read all kernel files into a single sources object ---
    for (const std::string& filePath : filePaths) {
        std::ifstream kernelFile(filePath);
        if (!kernelFile.is_open()) {
            throw std::runtime_error("Kernels not created. Reason: Could not open kernel file '" + filePath + "'");
        }
        sourceCodes.emplace_back((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
    }

    for (const auto& code : sourceCodes) {
        sources.push_back({code.c_str(), code.length()});
    }

    // --- Step 2: Create and build a SINGLE program from all sources ---
    cl::Program program(context, sources);
    cl_int build_err = program.build(devices);

    if (build_err != CL_SUCCESS) {
        std::string log_message = "Kernels not created. Reason: Program build failed. ";
        if (build_err == CL_BUILD_PROGRAM_FAILURE) {
            std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            log_message += "\n\n--- Build Log ---\n" + buildLog + "\n-----------------";
        } else {
            log_message += "Error code: " + std::to_string(build_err);
        }
        throw std::runtime_error(log_message);
    }

    // --- Step 3: Extract all kernels from the single compiled program ---
    cl_uint numKernels = 0;
    cl_int err = clCreateKernelsInProgram(program(), 0, NULL, &numKernels);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Kernels not created. Reason: Failed to get number of kernels from program. Error code: " + std::to_string(err));
    }
    if (numKernels == 0) {
        std::cerr << "Warning: No kernels found in any of the provided files. Skipping." << std::endl;
        return;
    }

    std::vector<cl_kernel> rawKernels(numKernels);
    err = clCreateKernelsInProgram(program(), numKernels, rawKernels.data(), NULL);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Kernels not created. Reason: Failed to create kernels from program. Error code: " + std::to_string(err));
    }

    // Populate the map
    for (cl_kernel rawKernel : rawKernels) {
        if (rawKernel == nullptr) {
            throw std::runtime_error("Kernels not created. Reason: OpenCL driver returned a null kernel handle.");
        }

        size_t nameSize;
        clGetKernelInfo(rawKernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &nameSize);
        std::vector<char> nameBuffer(nameSize);
        clGetKernelInfo(rawKernel, CL_KERNEL_FUNCTION_NAME, nameSize, nameBuffer.data(), NULL);
        std::string kernelName(nameBuffer.data());

        if (kernelMap.count(kernelName)) {
            std::cerr << "Warning: Duplicate kernel name '" << kernelName << "' found. Overwriting existing kernel." << std::endl;
        }
        kernelMap[kernelName] = cl::Kernel(rawKernel); // Use assignment to allow overwriting
    }
}

#endif // USE_CL