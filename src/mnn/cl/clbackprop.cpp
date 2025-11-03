#ifdef USE_OPENCL
#include "mnn.hpp"
#include <vector>
#include <stdexcept>


/**
 * @brief Backpropagation for mnn using OpenCL.
 * @param expected The expected output vector.
 */
void mnn::clBackprop(const std::vector<float>& expected) {
    // sizes
    size_t inputSize = input.size();
    
    // buffers
    cl::Buffer d_incoming, d_outgoing;
    std::vector<cl::Buffer> d_clayers(this->layers);
    std::vector<cl::Buffer> d_blayers(this->layers);
    std::vector<cl::Buffer> d_dotProds(this->layers);
    std::vector<cl::Buffer> d_activate(this->layers);

    // load values to buffers

    // kernels
    cl::Kernel kernelSigmoid;

    for(int i = 0; i < this->layers; i++) {
        // set kernel arguments
        // execute kernel
        // read back results
    }
}

void mnn::clBackprop(const std::vector<std::vector<float>>& expected) {
    // sizes
    size_t inputSize = input.size();

    // buffers
    cl::Buffer d_incoming, d_outgoing;
    std::vector<cl::Buffer> d_clayers(this->layers);
    std::vector<cl::Buffer> d_blayers(this->layers);
    std::vector<cl::Buffer> d_dotProds(this->layers);
    std::vector<cl::Buffer> d_activate(this->layers);

    // load values to buffers

    // kernels
    cl::Kernel kernelSigmoid;

    for(int i = 0; i < this->layers; i++) {
        // set kernel arguments
        // execute kernel
        // read back results
    }
}

/**
 * @brief Backpropagation for mnn2d using OpenCL.
 * @param expected The expected output vector.
 */
void mnn2d::clBackprop(const std::vector<float>& expected) {
    // sizes
    size_t inputSize = input.size();
    
    // buffers
    cl::Buffer d_incoming, d_outgoing;
    std::vector<cl::Buffer> d_clayers(this->layers);
    std::vector<cl::Buffer> d_blayers(this->layers);
    std::vector<cl::Buffer> d_dotProds(this->layers);
    std::vector<cl::Buffer> d_activate(this->layers);

    // load values to buffers

    // kernels
    cl::Kernel kernelSigmoid;

    for(int i = 0; i < this->layers; i++) {
        // set kernel arguments
        // execute kernel
        // read back results
    }
}

void mnn2d::clBackprop(const std::vector<std::vector<float>>& expected) {
    // sizes
    size_t inputSize = input.size();
    
    // buffers
    cl::Buffer d_incoming, d_outgoing;
    std::vector<cl::Buffer> d_clayers(this->layers);
    std::vector<cl::Buffer> d_blayers(this->layers);
    std::vector<cl::Buffer> d_dotProds(this->layers);
    std::vector<cl::Buffer> d_activate(this->layers);

    // load values to buffers

    // kernels
    cl::Kernel kernelSigmoid;

    for(int i = 0; i < this->layers; i++) {
        // set kernel arguments
        // execute kernel
        // read back results
    }
}

#endif