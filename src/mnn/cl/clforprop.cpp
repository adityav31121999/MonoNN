#ifdef USE_CL
#include "mnn.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <limits> // For std::numeric_limits

/**
 * @brief forprop for mnn using OpenCL
 * @param input input vector
 */
void mnn::clForprop(const std::vector<float>& input)
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
        throw std::runtime_error(std::string("Exception in mnn::clForprop: ") + e.what());
    }
}


/**
 * @brief forprop for mnn2d using OpenCL
 * @param input input matrix
 */
void mnn2d::clForprop(const std::vector<std::vector<float>>& input)
{
    try {
        // sizes
        cl_int err;
        cl::NDRange local_1d(WORKSIZE_1D);
        cl::NDRange local_2d(WORKSIZE_2DX, WORKSIZE_2DY);

        // buffers
        cl::Buffer d_input, d_output;
        std::vector<cl::Buffer> d_clayers(this->layers);
        std::vector<cl::Buffer> d_blayers(this->layers);
        std::vector<cl::Buffer> d_dotProds(this->layers);
        std::vector<cl::Buffer> d_activate(this->layers);

        // Create and write to input buffer
        d_input = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inHeight * inWidth, (void*)flatten(input).data(), &err); CL_CHECK(err);
        for(int i = 0; i < this->layers; i++) {
            size_t cweight_size = cweights[i].size() * cweights[i][0].size();
            size_t bweight_size = bweights[i].size() * bweights[i][0].size();
            d_clayers[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * cweight_size, (void*)flatten(cweights[i]).data(), &err); CL_CHECK(err);
            d_blayers[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * bweight_size, (void*)flatten(bweights[i]).data(), &err); CL_CHECK(err);
            size_t dotprod_size = dotProds[i].size() * dotProds[i][0].size();
            size_t activate_size = activate[i].size() * activate[i][0].size();
            d_dotProds[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * dotprod_size);
            d_activate[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * activate_size);
        }

        // kernels
        cl::Kernel kernelRelu, kernelForward, kernelMeanPool;
        kernelRelu = kernels.at("relu");
        kernelForward = kernels.at("kernelLayerForward4");
        kernelMeanPool = kernels.at("meanPool");

        // First layer
        int currentInWidth = inWidth, currentoutSize = width[0];
        kernelForward.setArg(0, d_input);
        kernelForward.setArg(1, d_dotProds[0]);
        kernelForward.setArg(2, d_clayers[0]);
        kernelForward.setArg(3, d_blayers[0]);
        kernelForward.setArg(4, inHeight);
        kernelForward.setArg(5, currentInWidth);
        kernelForward.setArg(6, currentoutSize);
        kernelForward.setArg(7, order);
        size_t local_dims[2] = {WORKSIZE_2DX, WORKSIZE_2DY};
        cl::NDRange globalForward = calculate_global_2d(local_dims, inHeight, currentoutSize);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForward, cl::NullRange, globalForward, local_2d));

        // Activation: Apply softmax to the flattened dot product
        size_t dotprod_size_layer0 = inHeight * width[0];

        kernelRelu.setArg(0, d_dotProds[0]);
        kernelRelu.setArg(1, d_activate[0]);
        kernelRelu.setArg(2, (int)dotprod_size_layer0);
        cl::NDRange localSoftmax(dotprod_size_layer0);
        cl::NDRange globalSoftmax(dotprod_size_layer0);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelRelu, cl::NullRange, globalSoftmax, localSoftmax));

        // copy results to host vectors
        size_t activate_size = activate[0].size() * activate[0][0].size();
        CL_CHECK(clCommandQueue.enqueueReadBuffer(d_dotProds[0], CL_TRUE, 0, sizeof(float) * activate_size, (void*)flatten(dotProds[0]).data()));
        CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[0], CL_TRUE, 0, sizeof(float) * activate_size, (void*)flatten(activate[0]).data()));

        // hidden layers
        for(int i = 1; i < this->layers; i++) {
            currentInWidth = width[i-1], currentoutSize = width[i];
            kernelForward.setArg(0, d_activate[i-1]);
            kernelForward.setArg(1, d_dotProds[i]);
            kernelForward.setArg(2, d_clayers[i]);
            kernelForward.setArg(3, d_blayers[i]);
            kernelForward.setArg(4, inHeight);
            kernelForward.setArg(5, currentInWidth);
            kernelForward.setArg(6, currentoutSize);
            kernelForward.setArg(7, order);
            cl::NDRange globalForward_i = calculate_global_2d(local_dims, inHeight, currentoutSize);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForward, cl::NullRange, globalForward_i, local_2d));

            // softmax to the flattened dot product
            size_t dotprod_size_layer_i = inHeight * width[i];

            kernelRelu.setArg(0, d_dotProds[i]);
            kernelRelu.setArg(1, d_activate[i]);
            kernelRelu.setArg(2, (int)dotprod_size_layer_i);
            cl::NDRange localSoftmax_i(dotprod_size_layer_i);
            cl::NDRange globalSoftmax_i(dotprod_size_layer_i);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelRelu, cl::NullRange, globalSoftmax_i, localSoftmax_i));

            activate_size = activate[i].size() * activate[i][0].size();
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_dotProds[i], CL_TRUE, 0, sizeof(float) * activate_size, (void*)flatten(dotProds[i]).data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[i], CL_TRUE, 0, sizeof(float) * activate_size, (void*)flatten(activate[i]).data()));
        }

        int last_layer_rows = inHeight, last_layer_cols = outSize;
        d_output = cl::Buffer(clContext, CL_MEM_WRITE_ONLY, sizeof(float) * last_layer_cols);
        kernelMeanPool.setArg(0, d_activate[layers - 1]);
        kernelMeanPool.setArg(1, d_output);
        kernelMeanPool.setArg(2, last_layer_rows);
        kernelMeanPool.setArg(3, last_layer_cols);
        kernelMeanPool.setArg(4, 1);
        cl::NDRange globalPool = calculate_global_1d(WORKSIZE_1D, last_layer_cols);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMeanPool, cl::NullRange, globalPool, local_1d));
        CL_CHECK(clCommandQueue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * output.size(), output.data()));
        output = softmax(output, SOFTMAX_TEMP);
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn2d::clForprop: ") + e.what());
    }
}

#endif