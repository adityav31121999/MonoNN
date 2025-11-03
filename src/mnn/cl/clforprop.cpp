#ifdef USE_OPENCL
#include <CL/cl.hpp>
#include "mnn.hpp"
#include <vector>
#include <stdexcept>


/**
 * @brief forprop for mnn using OpenCL
 * @param input input vector
 */
void mnn::clForprop(const std::vector<float>& input) {
    // sizes
    size_t inputSize = input.size();
    
    // buffers
    cl::Buffer d_input, d_output;
    std::vector<cl::Buffer> d_clayers(this->layers);
    std::vector<cl::Buffer> d_blayers(this->layers);
    std::vector<cl::Buffer> d_dotProds(this->layers);
    std::vector<cl::Buffer> d_activate(this->layers);

    // load values to buffers
    this->clCommandQueue.enqueueWriteBuffer(d_input, CL_TRUE, 0, sizeof(float) * inputSize, input.data());

    // kernels
    cl::Kernel kernelSigmoid;

    for(int i = 0; i < this->layers; i++) {
        // set kernel arguments
        // execute kernel
        // read back results
    }
}

/**
 * @brief forprop for mnn2d using OpenCL
 * @param input input matrix
 */
void mnn2d::clForprop(const std::vector<std::vector<float>>& input) {
    // sizes
    size_t inputSize = input.size();
    
    // buffers
    cl::Buffer d_input, d_output;
    std::vector<cl::Buffer> d_clayers(this->layers);
    std::vector<cl::Buffer> d_blayers(this->layers);
    std::vector<cl::Buffer> d_dotProds(this->layers);
    std::vector<cl::Buffer> d_activate(this->layers);

    // load values to buffers
    this->clCommandQueue.enqueueWriteBuffer(d_input, CL_TRUE, 0, sizeof(float) * inputSize, input.data());

    // kernels
    cl::Kernel kernelSigmoid;

    for(int i = 0; i < this->layers; i++) {
        // set kernel arguments
        // execute kernel
        // read back results
    }
}

#endif