#ifdef USE_CL
#include "mnn1d.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <limits> // For std::numeric_limits

/**
 * @brief forprop for mnn using OpenCL
 * @param input input vector
 */
void mnn1d::clForprop(const std::vector<float>& input)
{
    try {
        cl_int err;
        cl::NDRange local_1d(WORKSIZE_1D);

        // sizes
        size_t inputSize = input.size();

        // buffers
        cl::Buffer d_input, d_current_act;
        std::vector<cl::Buffer> d_clayers(this->layers);
        std::vector<cl::Buffer> d_blayers(this->layers);
        std::vector<cl::Buffer> d_dotProds(this->layers);
        std::vector<cl::Buffer> d_activate(this->layers);

        // Create and write to input buffer
        d_input = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputSize, (void*)input.data(), &err);
        CL_CHECK(err);

        // Create all other buffers and copy weight data
        for(int i = 0; i < this->layers; i++) {
            size_t cweight_size = cweights[i].size() * cweights[i][0].size();
            size_t bweight_size = bweights[i].size() * bweights[i][0].size();
            d_clayers[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * cweight_size, (void*)flatten(cweights[i]).data(), &err); CL_CHECK(err);
            d_blayers[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * bweight_size, (void*)flatten(bweights[i]).data(), &err); CL_CHECK(err);
            
            d_dotProds[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * dotProds[i].size());
            d_activate[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * activate[i].size());
        }

        // kernels
        cl::Kernel kernelSigmoid, kernelForward;
        kernelSigmoid = kernels.at("sigmoid");
        kernelForward = kernels.at("kernelLayerForward2");

        // first layer forward
        d_current_act = d_input;
        kernelForward.setArg(0, d_current_act);
        kernelForward.setArg(1, d_dotProds[0]);
        kernelForward.setArg(2, d_clayers[0]);
        kernelForward.setArg(3, d_blayers[0]);
        kernelForward.setArg(4, (int)inputSize);
        kernelForward.setArg(5, (int)dotProds[0].size());
        kernelForward.setArg(6, order);
        cl::NDRange globalForward = calculate_global_1d(WORKSIZE_1D, dotProds[0].size());
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForward, cl::NullRange, globalForward, local_1d));

        // activation
        kernelSigmoid.setArg(0, d_dotProds[0]);
        kernelSigmoid.setArg(1, d_activate[0]);
        kernelSigmoid.setArg(2, (int)dotProds[0].size());
        cl::NDRange globalSig = calculate_global_1d(WORKSIZE_1D, dotProds[0].size());
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSigmoid, cl::NullRange, globalSig, local_1d));

        // Copy results back to host vectors
        CL_CHECK(clCommandQueue.enqueueReadBuffer(d_dotProds[0], CL_TRUE, 0, sizeof(float) * dotProds[0].size(), dotProds[0].data()));
        CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[0], CL_TRUE, 0, sizeof(float) * activate[0].size(), activate[0].data()));
        // for second to last layer
        for(int i = 1; i < this->layers; i++) {
            // ith layer forward
            d_current_act = d_activate[i-1];
            kernelForward.setArg(0, d_current_act);
            kernelForward.setArg(1, d_dotProds[i]);
            kernelForward.setArg(2, d_clayers[i]);
            kernelForward.setArg(3, d_blayers[i]);
            kernelForward.setArg(4, (int)width[i-1]);
            kernelForward.setArg(5, (int)dotProds[i].size());
            kernelForward.setArg(6, order);
            globalForward = calculate_global_1d(WORKSIZE_1D, dotProds[i].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForward, cl::NullRange, globalForward, local_1d));

            // activate
            kernelSigmoid.setArg(0, d_dotProds[i]);
            kernelSigmoid.setArg(1, d_activate[i]);
            kernelSigmoid.setArg(2, (int)dotProds[i].size());
            globalSig = calculate_global_1d(WORKSIZE_1D, dotProds[i].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSigmoid, cl::NullRange, globalSig, local_1d));

            // Copy results back to host vectors
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_dotProds[i], CL_TRUE, 0, sizeof(float) * dotProds[i].size(), dotProds[i].data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[i], CL_TRUE, 0, sizeof(float) * activate[i].size(), activate[i].data()));
        }

        // Read the final activation back to the host output vector
        output = softmax(activate[layers - 1], SOFTMAX_TEMP);
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn1d::clForprop: ") + e.what());
    }
}

#endif