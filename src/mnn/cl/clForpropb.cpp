#ifdef USE_CL
#include "mnn.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <limits> // For std::numeric_limits

/**
 * @brief batch forprop for mnn using OpenCL
 * @param input input vector
 */
void mnn::clForprop(const std::vector<std::vector<float>>& input)
{
    try {
        this->batchSize = input.size();
        if (this->batchSize == 0) return;

        cl_int err;
        cl::NDRange local_1d(WORKSIZE_1D);

        // Flatten the batch input for a single transfer
        std::vector<float> flat_input;
        for(const auto& vec : input) {
            flat_input.insert(flat_input.end(), vec.begin(), vec.end());
        }
        size_t single_input_size = input[0].size();
        size_t total_input_size = flat_input.size();

        // buffers
        cl::Buffer d_input, d_current_act;
        std::vector<cl::Buffer> d_clayers(this->layers);
        std::vector<cl::Buffer> d_blayers(this->layers);
        std::vector<cl::Buffer> d_dotProds(this->layers);
        std::vector<cl::Buffer> d_activate(this->layers);

        // Create and write to input buffer
        d_input = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * total_input_size, (void*)flat_input.data(), &err); CL_CHECK(err);

        // Create all other buffers and copy weight data
        for(int i = 0; i < this->layers; i++) {
            size_t cweight_size = cweights[i].size() * cweights[i][0].size();
            size_t bweight_size = bweights[i].size() * bweights[i][0].size();
            d_clayers[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * cweight_size, (void*)flatten(cweights[i]).data());
            d_blayers[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * bweight_size, (void*)flatten(bweights[i]).data());
            
            size_t layer_output_size = batchSize * width[i];
            d_dotProds[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * layer_output_size);
            d_activate[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * layer_output_size);
        }

        // kernels
        cl::Kernel kernelSigmoid, kernelForward;
        kernelSigmoid = kernels.at("sigmoid");
        kernelForward = kernels.at("kernelLayerForwardBatch2");

        // first layer forward
        d_current_act = d_input;
        kernelForward.setArg(0, d_current_act);
        kernelForward.setArg(1, d_dotProds[0]);
        kernelForward.setArg(2, d_clayers[0]);
        kernelForward.setArg(3, d_blayers[0]);
        kernelForward.setArg(4, (int)batchSize);
        kernelForward.setArg(5, (int)single_input_size);
        kernelForward.setArg(6, (int)width[0]);
        kernelForward.setArg(7, order);
        size_t global_work_size_2d[2] = {(size_t)batchSize, (size_t)width[0]};
        cl::NDRange globalForward(global_work_size_2d[0], global_work_size_2d[1]);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForward, cl::NullRange, globalForward, cl::NullRange));

        // activation
        kernelSigmoid.setArg(0, d_dotProds[0]);
        kernelSigmoid.setArg(1, d_activate[0]);
        kernelSigmoid.setArg(2, (int)(batchSize * width[0]));
        cl::NDRange globalSig = calculate_global_1d(WORKSIZE_1D, batchSize * width[0]);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSigmoid, cl::NullRange, globalSig, local_1d));

        // for second to last layer
        for(int i = 1; i < this->layers; i++) {
            d_current_act = d_activate[i-1];
            // ith layer forward
            kernelForward.setArg(0, d_current_act);
            kernelForward.setArg(1, d_dotProds[i]);
            kernelForward.setArg(2, d_clayers[i]);
            kernelForward.setArg(3, d_blayers[i]);
            kernelForward.setArg(4, (int)batchSize);
            kernelForward.setArg(5, (int)width[i-1]);
            kernelForward.setArg(6, (int)width[i]);
            kernelForward.setArg(7, order);
            global_work_size_2d[0] = (size_t)batchSize;
            global_work_size_2d[1] = (size_t)width[i];
            globalForward = cl::NDRange(global_work_size_2d[0], global_work_size_2d[1]);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForward, cl::NullRange, globalForward, cl::NullRange));

            // activate
            kernelSigmoid.setArg(0, d_dotProds[i]);
            kernelSigmoid.setArg(1, d_activate[i]);
            kernelSigmoid.setArg(2, (int)(batchSize * width[i]));
            globalSig = calculate_global_1d(WORKSIZE_1D, batchSize * width[i]);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSigmoid, cl::NullRange, globalSig, local_1d));
        }

        // Read the final activation and other results back to the host
        std::vector<float> final_activations(batchSize * outSize);
        CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[layers - 1], CL_TRUE, 0, sizeof(float) * final_activations.size(), final_activations.data()));

        for(int i = 0; i < batchSize; ++i) {
            std::copy(final_activations.begin() + i * outSize, final_activations.begin() + (i + 1) * outSize, outputBatch[i].begin());
        }

        // copy dot and acts
        for(int i=0; i<layers; ++i) {
            std::vector<float> dot_flat(batchSize * width[i]);
            std::vector<float> act_flat(batchSize * width[i]);
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_dotProds[i], CL_TRUE, 0, sizeof(float) * dot_flat.size(), dot_flat.data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[i], CL_TRUE, 0, sizeof(float) * act_flat.size(), act_flat.data()));
            for(int j=0; j<batchSize; ++j) {
                std::copy(dot_flat.begin() + j * width[i], dot_flat.begin() + (j+1) * width[i], dotBatch[i][j].begin());
                std::copy(act_flat.begin() + j * width[i], act_flat.begin() + (j+1) * width[i], actBatch[i][j].begin());
            }
        }
        outputBatch = actBatch[layers-1];
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn::clForprop: ") + e.what());
    }
}


/**
 * @brief forprop for mnn2d using OpenCL
 * @param input input matrix
 */
void mnn2d::clForprop(const std::vector<std::vector<std::vector<float>>>& input)
{
    try {
        this->batchSize = input.size();
        if (this->batchSize == 0) return;

        // sizes
        cl_int err;
        cl::NDRange local_1d(WORKSIZE_1D);
        cl::NDRange local_2d(WORKSIZE_2DX, WORKSIZE_2DY);

        std::vector<float> flat_input;
        for(const auto& mat : input) {
            for(const auto& row : mat) {
                flat_input.insert(flat_input.end(), row.begin(), row.end());
            }
        }
        size_t single_input_height = input[0].size();
        size_t single_input_width = input[0][0].size();
        size_t total_input_size = flat_input.size();

        // buffers
        cl::Buffer d_input, d_current_act;
        std::vector<cl::Buffer> d_clayers(this->layers);
        std::vector<cl::Buffer> d_blayers(this->layers);
        std::vector<cl::Buffer> d_dotProds(this->layers);
        std::vector<cl::Buffer> d_activate(this->layers);

        // Create and write to input buffer
        d_input = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * total_input_size, (void*)flat_input.data(), &err); CL_CHECK(err);

        for(int i = 0; i < this->layers; i++) {
            size_t cweight_size = cweights[i].size() * cweights[i][0].size();
            size_t bweight_size = bweights[i].size() * bweights[i][0].size();
            d_clayers[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * cweight_size, (void*)flatten(cweights[i]).data());
            d_blayers[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * bweight_size, (void*)flatten(bweights[i]).data());
            
            size_t layer_output_size = batchSize * inHeight * width[i];
            d_dotProds[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * layer_output_size);
            d_activate[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * layer_output_size);
        }

        // kernels
        cl::Kernel kernelSoftMax, kernelForward, kernelMeanPool;
        kernelSoftMax = kernels.at("softmax");
        kernelForward = kernels.at("kernelLayerForwardBatch4");
        kernelMeanPool = kernels.at("meanPool");

        // First layer
        d_current_act = d_input;
        kernelForward.setArg(0, d_current_act);
        kernelForward.setArg(1, d_dotProds[0]);
        kernelForward.setArg(2, d_clayers[0]);
        kernelForward.setArg(3, d_blayers[0]);
        kernelForward.setArg(4, (int)batchSize);
        kernelForward.setArg(5, (int)single_input_height);
        kernelForward.setArg(6, (int)single_input_width);
        kernelForward.setArg(7, (int)width[0]);
        kernelForward.setArg(8, order);
        cl::NDRange globalForward(batchSize, single_input_height, width[0]);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForward, cl::NullRange, globalForward, cl::NullRange));

        // Activation for first layer
        size_t dotprod_size_layer0 = batchSize * inHeight * width[0];
        kernelSoftMax.setArg(0, d_dotProds[0]);
        kernelSoftMax.setArg(1, d_activate[0]);
        kernelSoftMax.setArg(2, SOFTMAX_TEMP);
        kernelSoftMax.setArg(3, (int)dotprod_size_layer0);
        cl::NDRange globalSoftmax = calculate_global_1d(WORKSIZE_1D, dotprod_size_layer0);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSoftMax, cl::NullRange, globalSoftmax, local_1d));

        // hidden layers
        for(int i = 1; i < this->layers; i++) {
            d_current_act = d_activate[i-1];
            kernelForward.setArg(0, d_current_act);
            kernelForward.setArg(1, d_dotProds[i]);
            kernelForward.setArg(2, d_clayers[i]);
            kernelForward.setArg(3, d_blayers[i]);
            kernelForward.setArg(4, (int)batchSize);
            kernelForward.setArg(5, (int)inHeight);
            kernelForward.setArg(6, (int)width[i-1]);
            kernelForward.setArg(7, (int)width[i]);
            kernelForward.setArg(8, order);
            globalForward = cl::NDRange(batchSize, inHeight, width[i]);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForward, cl::NullRange, globalForward, cl::NullRange));

            // softmax to the flattened dot product
            size_t dotprod_size_layer_i = batchSize * inHeight * width[i];
            kernelSoftMax.setArg(0, d_dotProds[i]);
            kernelSoftMax.setArg(1, d_activate[i]);
            kernelSoftMax.setArg(2, SOFTMAX_TEMP);
            kernelSoftMax.setArg(3, (int)dotprod_size_layer_i);
            globalSoftmax = calculate_global_1d(WORKSIZE_1D, dotprod_size_layer_i);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSoftMax, cl::NullRange, globalSoftmax, local_1d));
        }

        // Mean pool the final activation layer
        cl::Buffer d_final_output(clContext, CL_MEM_WRITE_ONLY, sizeof(float) * batchSize * outWidth);
        kernelMeanPool.setArg(0, d_activate[layers-1]);
        kernelMeanPool.setArg(1, d_final_output);
        kernelMeanPool.setArg(2, inHeight); // rows to pool over
        kernelMeanPool.setArg(3, outWidth);
        kernelMeanPool.setArg(4, batchSize);
        cl::NDRange globalPool(batchSize, outWidth);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMeanPool, cl::NullRange, globalPool, cl::NullRange));
        // Read final output and intermediate results back to host
        std::vector<float> final_output_flat(batchSize * outWidth);
        CL_CHECK(clCommandQueue.enqueueReadBuffer(d_final_output, CL_TRUE, 0, sizeof(float) * final_output_flat.size(), final_output_flat.data()));
        for(int i=0; i<batchSize; ++i) {
            std::copy(final_output_flat.begin() + i * outWidth, final_output_flat.begin() + (i+1) * outWidth, outputBatch[i].begin());
        }
        outputBatch = reshape(final_output_flat, batchSize, outWidth);

        // copy dot and acts
        for(int i=0; i<layers; ++i) {
            size_t layer_size = batchSize * inHeight * width[i];
            std::vector<float> dot_flat(layer_size);
            std::vector<float> act_flat(layer_size);
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_dotProds[i], CL_TRUE, 0, sizeof(float) * layer_size, dot_flat.data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[i], CL_TRUE, 0, sizeof(float) * layer_size, act_flat.data()));

            for(int j=0; j<batchSize; ++j) {
                size_t single_item_size = inHeight * width[i];
                std::vector<float> single_dot(dot_flat.begin() + j * single_item_size, dot_flat.begin() + (j+1) * single_item_size);
                std::vector<float> single_act(act_flat.begin() + j * single_item_size, act_flat.begin() + (j+1) * single_item_size);
                dotBatch[i][j] = reshape(single_dot, inHeight, width[i]);
                actBatch[i][j] = reshape(single_act, inHeight, width[i]);
            }
        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn2d::clForprop: ") + e.what());
    }
}

#endif