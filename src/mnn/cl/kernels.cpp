#ifdef USE_OPENCL
// In a new file, e.g., d:\monoNN\src\operators.cpp
#include "operators.hpp" // Use the correct path to your header
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <map>
#include <iostream>
#include <CL/cl.hpp>

// Define and initialize the global kernelNames variable here.
const std::vector<std::string> kernelNames = {
    // actvations and derivative
    "sigmoid",
    "sigmoidDer",
    "softmax",
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
 * This is a highly defensive version designed to prevent runtime aborts by avoiding
 * temporary C++ wrapper objects and using the safest possible C-API interaction.
 * @param context A valid, initialized cl::Context object.
 * @param filePath The path to the .cl kernel file.
 * @param kernels A reference to a std::map to be populated. The map will be
 *                  cleared before being filled.
 * @throws std::runtime_error if any step of the process fails.
 */
void createKernelsFromFile(const cl::Context& context, const std::string& filePath, std::map<std::string, cl::Kernel>& kernelMap) {
    kernelMap.clear();

    // Read kernel file
    std::ifstream kernelFile(filePath);
    if (!kernelFile.is_open()) {
        throw std::runtime_error("Kernels not created. Reason: Could not open kernel file '" + filePath + "'");
    }
    std::string kernelCode((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
    if (kernelCode.empty()) {
        throw std::runtime_error("Kernels not created. Reason: Kernel file is empty '" + filePath + "'");
    }

    // Create and build the OpenCL program
    cl::Program::Sources sources;
    sources.push_back({kernelCode.c_str(), kernelCode.length()});
    cl::Program program(context, sources);
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl_int build_err = program.build(devices);

    if (build_err != CL_SUCCESS) {
        std::string log_message = "Kernels not created. Reason: Program build failed. ";
        if (build_err == CL_BUILD_PROGRAM_FAILURE) {
            std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            log_message += "\n\n--- Build Log ---\n" + buildLog + "\n-----------------";
        } else {
            log_message += "Error code: " + std::to_string(build_err) + " (" + oclErrorString(build_err) + ")";
        }
        throw std::runtime_error(log_message);
    }

    // Create raw C-style kernel handles
    cl_uint numKernels = 0;
    CL_CHECK(clCreateKernelsInProgram(program(), 0, NULL, &numKernels));
    if (numKernels == 0) return;

    std::vector<cl_kernel> rawKernels(numKernels);
    CL_CHECK(clCreateKernelsInProgram(program(), numKernels, rawKernels.data(), NULL));

    // populate the map
    for (cl_kernel rawKernel : rawKernels) {
        if (rawKernel == nullptr) {
            throw std::runtime_error("Kernels not created. Reason: OpenCL driver returned a null kernel handle.");
        }

        size_t nameSize;
        CL_CHECK(clGetKernelInfo(rawKernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &nameSize));
        std::vector<char> nameBuffer(nameSize);
        CL_CHECK(clGetKernelInfo(rawKernel, CL_KERNEL_FUNCTION_NAME, nameSize, nameBuffer.data(), NULL));
        std::string kernelName(nameBuffer.data());
        kernelMap.emplace(kernelName, cl::Kernel(rawKernel));
    }
}

#endif // USE_OPENCL