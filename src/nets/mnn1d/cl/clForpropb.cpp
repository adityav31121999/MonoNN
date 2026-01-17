#ifdef USE_CL
#include "mnn1d.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <limits> // For std::numeric_limits

/**
 * @brief batch forprop for mnn using OpenCL
 * @param input input vector
 */
void mnn1d::clForprop(const std::vector<std::vector<float>>& input)
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
        kernelSigmoid = kernels.at("relu");
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
        for(int i = 0; i < batchSize; ++i) {
            outputBatch[i] = softmax(outputBatch[i], SOFTMAX_TEMP);
        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn1d::clForprop: ") + e.what());
    }
}

#endif