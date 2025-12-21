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
        CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[layers - 1], CL_TRUE, 0, sizeof(float) * output.size(), output.data()));
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
        cl::Kernel kernelSoftMax, kernelForward, kernelMeanPool;
        kernelSoftMax = kernels.at("softmax");
        kernelForward = kernels.at("kernelLayerForward4");
        kernelMeanPool = kernels.at("meanPool");

        // First layer
        int currentInWidth = inWidth, currentOutWidth = width[0];
        kernelForward.setArg(0, d_input);
        kernelForward.setArg(1, d_dotProds[0]);
        kernelForward.setArg(2, d_clayers[0]);
        kernelForward.setArg(3, d_blayers[0]);
        kernelForward.setArg(4, inHeight);
        kernelForward.setArg(5, currentInWidth);
        kernelForward.setArg(6, currentOutWidth);
        kernelForward.setArg(7, order);
        size_t local_dims[2] = {WORKSIZE_2DX, WORKSIZE_2DY};
        cl::NDRange globalForward = calculate_global_2d(local_dims, inHeight, currentOutWidth);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForward, cl::NullRange, globalForward, local_2d));

        // Activation: Apply softmax to the flattened dot product
        size_t dotprod_size_layer0 = inHeight * width[0];

        if (dotprod_size_layer0 <= WORKSIZE_1D) { // Use the single-work-group softmax if size permits
            kernelSoftMax.setArg(0, d_dotProds[0]);
            kernelSoftMax.setArg(1, d_activate[0]);
            kernelSoftMax.setArg(2, SOFTMAX_TEMP);
            kernelSoftMax.setArg(3, (int)dotprod_size_layer0);
            cl::NDRange localSoftmax(dotprod_size_layer0);
            cl::NDRange globalSoftmax(dotprod_size_layer0);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSoftMax, cl::NullRange, globalSoftmax, localSoftmax));
        }
        else {
            // Get kernels
            cl::Kernel kernelSoftMaxReduce = kernels.at("softmax_reduce");
            cl::Kernel kernelSoftMaxNormalize = kernels.at("softmax_normalize");
            // number of work-groups
            size_t num_work_groups = (dotprod_size_layer0 + WORKSIZE_1D - 1) / WORKSIZE_1D;
            size_t partial_results_buffer_size = num_work_groups * 2;
            // allocate intermediate buffer for partial results
            cl::Buffer d_partial_results(clContext, CL_MEM_READ_WRITE, sizeof(float) * partial_results_buffer_size, nullptr, &err); CL_CHECK(err);

            // launch softmax_reduce kernel
            kernelSoftMaxReduce.setArg(0, d_dotProds[0]);
            kernelSoftMaxReduce.setArg(1, d_partial_results);
            kernelSoftMaxReduce.setArg(2, cl::Local(sizeof(float) * WORKSIZE_1D * 2));
            kernelSoftMaxReduce.setArg(3, (int)dotprod_size_layer0);
            kernelSoftMaxReduce.setArg(4, SOFTMAX_TEMP);
            cl::NDRange globalReduce = calculate_global_1d(WORKSIZE_1D, dotprod_size_layer0);
            cl::NDRange localReduce(WORKSIZE_1D);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSoftMaxReduce, cl::NullRange, globalReduce, localReduce));

            // read partial results back to host
            std::vector<float> h_partial_results(partial_results_buffer_size);
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_partial_results, CL_TRUE, 0, sizeof(float) * partial_results_buffer_size, h_partial_results.data()));
            // compute global max and global sum on host
            float global_max = -(std::numeric_limits<float>::max)();
            float global_sum = 0.0f;
            for (size_t k = 0; k < num_work_groups; ++k) {
                global_sum += h_partial_results[2 * k];
                global_max = (std::max)(global_max, h_partial_results[2 * k + 1]);
            }

            // launch softmax_normalize kernel
            kernelSoftMaxNormalize.setArg(0, d_dotProds[0]);
            kernelSoftMaxNormalize.setArg(1, d_activate[0]);
            kernelSoftMaxNormalize.setArg(2, (int)dotprod_size_layer0);
            kernelSoftMaxNormalize.setArg(3, SOFTMAX_TEMP);
            kernelSoftMaxNormalize.setArg(4, global_max);
            kernelSoftMaxNormalize.setArg(5, global_sum);
            cl::NDRange globalNormalize = calculate_global_1d(WORKSIZE_1D, dotprod_size_layer0);
            cl::NDRange localNormalize(WORKSIZE_1D);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSoftMaxNormalize, cl::NullRange, globalNormalize, localNormalize));
        }

        // copy results to host vectors
        size_t activate_size = activate[0].size() * activate[0][0].size();
        CL_CHECK(clCommandQueue.enqueueReadBuffer(d_dotProds[0], CL_TRUE, 0, sizeof(float) * activate_size, (void*)flatten(dotProds[0]).data()));
        CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[0], CL_TRUE, 0, sizeof(float) * activate_size, (void*)flatten(activate[0]).data()));

        // hidden layers
        for(int i = 1; i < this->layers; i++) {
            currentInWidth = width[i-1], currentOutWidth = width[i];
            kernelForward.setArg(0, d_activate[i-1]);
            kernelForward.setArg(1, d_dotProds[i]);
            kernelForward.setArg(2, d_clayers[i]);
            kernelForward.setArg(3, d_blayers[i]);
            kernelForward.setArg(4, inHeight);
            kernelForward.setArg(5, currentInWidth);
            kernelForward.setArg(6, currentOutWidth);
            kernelForward.setArg(7, order);
            cl::NDRange globalForward_i = calculate_global_2d(local_dims, inHeight, currentOutWidth);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForward, cl::NullRange, globalForward_i, local_2d));

            // softmax to the flattened dot product
            size_t dotprod_size_layer_i = inHeight * width[i];

            if (dotprod_size_layer_i <= WORKSIZE_1D) { // Use the single-work-group softmax if size permits
                kernelSoftMax.setArg(0, d_dotProds[i]);
                kernelSoftMax.setArg(1, d_activate[i]);
                kernelSoftMax.setArg(2, SOFTMAX_TEMP);
                kernelSoftMax.setArg(3, (int)dotprod_size_layer_i);
                cl::NDRange localSoftmax_i(dotprod_size_layer_i);
                cl::NDRange globalSoftmax_i(dotprod_size_layer_i);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSoftMax, cl::NullRange, globalSoftmax_i, localSoftmax_i));
            }
            else {
                cl::Kernel kernelSoftMaxReduce = kernels.at("softmax_reduce");
                cl::Kernel kernelSoftMaxNormalize = kernels.at("softmax_normalize");
                // calculate number of work-groups
                size_t num_work_groups = (dotprod_size_layer_i + WORKSIZE_1D - 1) / WORKSIZE_1D;
                size_t partial_results_buffer_size = num_work_groups * 2; // Each work-group writes max and sum
                // allocate intermediate buffer for partial results
                cl::Buffer d_partial_results(clContext, CL_MEM_READ_WRITE, sizeof(float) * partial_results_buffer_size, nullptr, &err); CL_CHECK(err);

                // launch softmax_reduce kernel
                kernelSoftMaxReduce.setArg(0, d_dotProds[i]);
                kernelSoftMaxReduce.setArg(1, d_partial_results);
                kernelSoftMaxReduce.setArg(2, cl::Local(sizeof(float) * WORKSIZE_1D * 2));
                kernelSoftMaxReduce.setArg(3, (int)dotprod_size_layer_i);
                kernelSoftMaxReduce.setArg(4, SOFTMAX_TEMP);
                cl::NDRange globalReduce = calculate_global_1d(WORKSIZE_1D, dotprod_size_layer_i);
                cl::NDRange localReduce(WORKSIZE_1D);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSoftMaxReduce, cl::NullRange, globalReduce, localReduce));

                // read partial results back to host
                std::vector<float> h_partial_results(partial_results_buffer_size);
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_partial_results, CL_TRUE, 0, sizeof(float) * partial_results_buffer_size, h_partial_results.data()));
                // compute global max and global sum on host
                float global_max = -(std::numeric_limits<float>::max)();
                float global_sum = 0.0f;
                for (size_t k = 0; k < num_work_groups; ++k) {
                    global_sum += h_partial_results[2 * k];
                    global_max = (std::max)(global_max, h_partial_results[2 * k + 1]);
                }

                // launch softmax_normalize kernel
                kernelSoftMaxNormalize.setArg(0, d_dotProds[i]);
                kernelSoftMaxNormalize.setArg(1, d_activate[i]);
                kernelSoftMaxNormalize.setArg(2, (int)dotprod_size_layer_i);
                kernelSoftMaxNormalize.setArg(3, SOFTMAX_TEMP);
                kernelSoftMaxNormalize.setArg(4, global_max);
                kernelSoftMaxNormalize.setArg(5, global_sum);
                cl::NDRange globalNormalize = calculate_global_1d(WORKSIZE_1D, dotprod_size_layer_i);
                cl::NDRange localNormalize(WORKSIZE_1D);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSoftMaxNormalize, cl::NullRange, globalNormalize, localNormalize));
            }
            activate_size = activate[i].size() * activate[i][0].size();
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_dotProds[i], CL_TRUE, 0, sizeof(float) * activate_size, (void*)flatten(dotProds[i]).data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[i], CL_TRUE, 0, sizeof(float) * activate_size, (void*)flatten(activate[i]).data()));
        }

        int last_layer_rows = inHeight, last_layer_cols = outWidth;
        d_output = cl::Buffer(clContext, CL_MEM_WRITE_ONLY, sizeof(float) * last_layer_cols);
        kernelMeanPool.setArg(0, d_activate[layers - 1]);
        kernelMeanPool.setArg(1, d_output);
        kernelMeanPool.setArg(2, last_layer_rows);
        kernelMeanPool.setArg(3, last_layer_cols);
        kernelMeanPool.setArg(4, 1);
        cl::NDRange globalPool = calculate_global_1d(WORKSIZE_1D, last_layer_cols);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMeanPool, cl::NullRange, globalPool, local_1d));
        CL_CHECK(clCommandQueue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * output.size(), output.data()));
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn2d::clForprop: ") + e.what());
    }
}

#endif