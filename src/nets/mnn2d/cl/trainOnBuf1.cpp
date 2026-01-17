#ifdef USE_CL
#include "mnn1d.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm> // For std::max
#include <cmath>     // For std::ceil
#include <limits>    // For std::numeric_limits

/**
 * @brief Trains the mnn2d network on a single input-target pair with optimized buffer management.
 * @param input The input matrix.
 * @param target The target vector.
 */
void mnn2d::clBufTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target) {
    cl_int err;
    cl::NDRange local_1d(WORKSIZE_1D);
    cl::NDRange local_2d(WORKSIZE_2DX, WORKSIZE_2DY);
    size_t size2d[2] = {WORKSIZE_2DX, WORKSIZE_2DY};

    // --- Buffer Allocation ---
    cl::Buffer d_in, d_exp, d_out, d_err;
    std::vector<cl::Buffer> d_cweights(layers), d_bweights(layers), d_gradC(layers), d_gradB(layers);
    std::vector<cl::Buffer> d_dotProds(layers), d_activate(layers), d_incoming(layers);
    std::vector<cl::Buffer> d_dpow(layers > 1 ? layers - 1 : 0), d_dact(layers > 1 ? layers - 1 : 0);
    cl::Buffer d_grad_x_CT_buf, d_dprev_p_buf, d_dprev_act_buf;
    cl::Buffer d_final_output;
    cl::Buffer d_C_T_buf, d_prev_p_T_buf, d_ones_all_buf, d_partial_results_buf;

    try {
        std::vector<float> flat_input = flatten(input);
        d_in = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, flat_input.size() * sizeof(float), (void*)flat_input.data(), &err); CL_CHECK(err);
        d_exp = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, target.size() * sizeof(float), (void*)target.data(), &err); CL_CHECK(err);
        d_out = cl::Buffer(clContext, CL_MEM_READ_WRITE, output.size() * sizeof(float)); CL_CHECK(err);
        d_err = cl::Buffer(clContext, CL_MEM_READ_WRITE, output.size() * sizeof(float)); CL_CHECK(err);

        size_t max_act_size = 0;
        for (int i = 0; i < layers; ++i) max_act_size = std::max(max_act_size, activate[i].size() * activate[i][0].size());
        max_act_size = std::max(max_act_size, (size_t)inHeight * inWidth);
        d_grad_x_CT_buf = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_act_size); CL_CHECK(err);
        d_dprev_p_buf = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_act_size); CL_CHECK(err);
        d_prev_p_T_buf = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_act_size); CL_CHECK(err);
        d_dprev_act_buf = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_act_size); CL_CHECK(err);
        d_final_output = cl::Buffer(clContext, CL_MEM_WRITE_ONLY, sizeof(float) * outSize); CL_CHECK(err);
        d_partial_results_buf = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_act_size); CL_CHECK(err);

        std::vector<float> all_ones(max_act_size, 1.0f);
        d_ones_all_buf = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * max_act_size, all_ones.data(), &err); CL_CHECK(err);

        size_t max_weight_size = 0;
        for(const auto& w : cweights) max_weight_size = std::max(max_weight_size, w.size() * w[0].size());
        d_C_T_buf = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_weight_size); CL_CHECK(err);

        for (int i = 0; i < layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            size_t act_size = activate[i].size() * activate[i][0].size();
            d_cweights[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, c_size * sizeof(float)); CL_CHECK(err);
            d_bweights[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, b_size * sizeof(float)); CL_CHECK(err);
            d_gradC[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, c_size * sizeof(float)); CL_CHECK(err);
            d_gradB[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, b_size * sizeof(float)); CL_CHECK(err);
            d_dotProds[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
            d_activate[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
            d_incoming[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
            if (i < layers - 1) {
                d_dpow[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
                d_dact[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
            }
        }

        int epoch = 0;
        float initialLR = this->learningRate;
        int maxEpochs = 2000;

        // --- Training Loop ---
        while (epoch < maxEpochs) {
            for (int i = 0; i < layers; ++i) {
                std::vector<float> flat_c = flatten(cweights[i]);
                std::vector<float> flat_b = flatten(bweights[i]);
                CL_CHECK(clCommandQueue.enqueueWriteBuffer(d_cweights[i], CL_TRUE, 0, flat_c.size() * sizeof(float), flat_c.data()));
                CL_CHECK(clCommandQueue.enqueueWriteBuffer(d_bweights[i], CL_TRUE, 0, flat_b.size() * sizeof(float), flat_b.data()));
            }

            // --- Forward Propagation (adapted from mnn2d::clForprop) ---
            cl::Buffer d_current_act = d_in;
            cl::Kernel kernelForward = kernels.at("kernelLayerForward4");
            cl::Kernel kernelSoftMax = kernels.at("softmax");
            cl::Kernel kernelMeanPool = kernels.at("meanPool");

            // First layer
            int currentInHeight = inHeight, currentInWidth = inWidth, currentoutSize = width[0];
            kernelForward.setArg(0, d_current_act);
            kernelForward.setArg(1, d_dotProds[0]);
            kernelForward.setArg(2, d_cweights[0]);
            kernelForward.setArg(3, d_bweights[0]);
            kernelForward.setArg(4, currentInHeight);
            kernelForward.setArg(5, currentInWidth);
            kernelForward.setArg(6, currentoutSize);
            kernelForward.setArg(7, order);
            cl::NDRange globalForward = calculate_global_2d(size2d, currentInHeight, currentoutSize);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForward, cl::NullRange, globalForward, local_2d));

            size_t dotprod_size_layer0 = inHeight * width[0];
            kernelSoftMax.setArg(0, d_dotProds[0]);
            kernelSoftMax.setArg(1, d_activate[0]);
            kernelSoftMax.setArg(2, SOFTMAX_TEMP);
            kernelSoftMax.setArg(3, (int)dotprod_size_layer0);
            cl::NDRange globalSoftmax = calculate_global_1d(WORKSIZE_1D, dotprod_size_layer0);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSoftMax, cl::NullRange, globalSoftmax, local_1d));

            // Hidden layers
            for (int i = 1; i < layers; ++i) {
                d_current_act = d_activate[i - 1];
                currentInWidth = width[i - 1];
                currentoutSize = width[i];
                kernelForward.setArg(0, d_current_act);
                kernelForward.setArg(1, d_dotProds[i]);
                kernelForward.setArg(2, d_cweights[i]);
                kernelForward.setArg(3, d_bweights[i]);
                kernelForward.setArg(4, inHeight);
                kernelForward.setArg(5, currentInWidth);
                kernelForward.setArg(6, currentoutSize);
                kernelForward.setArg(7, order);
                globalForward = calculate_global_2d(size2d, inHeight, currentoutSize);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForward, cl::NullRange, globalForward, local_2d));

                size_t dotprod_size_layer_i = inHeight * width[i];
                kernelSoftMax.setArg(0, d_dotProds[i]);
                kernelSoftMax.setArg(1, d_activate[i]);
                kernelSoftMax.setArg(2, SOFTMAX_TEMP);
                kernelSoftMax.setArg(3, (int)dotprod_size_layer_i);
                globalSoftmax = calculate_global_1d(WORKSIZE_1D, dotprod_size_layer_i);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSoftMax, cl::NullRange, globalSoftmax, local_1d));
            }

            // Mean pool the final activation layer
            kernelMeanPool.setArg(0, d_activate[layers - 1]);
            kernelMeanPool.setArg(1, d_final_output);
            kernelMeanPool.setArg(2, inHeight);
            kernelMeanPool.setArg(3, outSize);
            kernelMeanPool.setArg(4, 1);
            cl::NDRange globalPool = calculate_global_1d(WORKSIZE_1D, outSize);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMeanPool, cl::NullRange, globalPool, local_1d));
            CL_CHECK(clCommandQueue.finish());

            // Copy output D2H to check for correctness and loss
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_final_output, CL_TRUE, 0, output.size() * sizeof(float), output.data()));
            output = softmax(output, SOFTMAX_TEMP);

            if (maxIndex(output) == maxIndex(target)) {
                std::cout << "Correct output predicted at epoch " << epoch << "." << std::endl;
                break;
            }
            epoch++;
            currloss = crossEntropy(output, target);
            std::cout << "Current CE Loss at epoch " << epoch << " : " << currloss << std::endl;

            // --- Backward Propagation (adapted from mnn2d::clBackprop) ---
            cl::Kernel kernelSub = kernels.at("subtract");
            cl::Kernel KernelReluDer = kernels.at("reluDer");
            cl::Kernel kernelHadamard2 = kernels.at("hadamard2");
            cl::Kernel kernelMatMul = kernels.at("matxmat2mat");
            cl::Kernel kernelScale = kernels.at("scaleByValue");
            
            cl::Kernel kernelUpdateWeights;
            switch (weightUpdateType) {
                case 0: kernelUpdateWeights = kernels.at("kernelUpdateWeights"); break;
                case 1: kernelUpdateWeights = kernels.at("kernelUpdateWeightsWithL1"); break;
                case 2: kernelUpdateWeights = kernels.at("kernelUpdateWeightsWithL2"); break;
                case 3: kernelUpdateWeights = kernels.at("kernelUpdateWeightsElasticNet"); break;
                case 4: kernelUpdateWeights = kernels.at("kernelUpdateWeightsWithWeightDecay"); break;
                case 5: kernelUpdateWeights = kernels.at("kernelUpdateWeightsDropout"); break;
                default: throw std::runtime_error("Invalid weight update type");
            }
            cl::Kernel kernelTranspose = kernels.at("transpose");
            cl::Kernel kernelDPow = kernels.at("dPower");
            cl::Kernel kernelPower = kernels.at("power");

            // Initial error (output - expected)
            kernelSub.setArg(0, d_out);
            kernelSub.setArg(1, d_exp);
            kernelSub.setArg(2, d_err);
            kernelSub.setArg(3, (int)output.size());
            cl::NDRange globalSub = calculate_global_1d(WORKSIZE_1D, output.size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSub, cl::NullRange, globalSub, local_1d));

            // Distribute the error from d_err to each row of the last layer's incoming gradient buffer.
            for (size_t r = 0; r < activate[layers - 1].size(); ++r) {
                CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_err, d_incoming[layers - 1], 0, r * output.size() * sizeof(float), sizeof(float) * output.size()));
            }

            // Backpropagation from last to second layer
            for (int layer = layers - 1; layer >= 1; --layer) {
                int prev_rows = activate[layer - 1].size();
                int prev_cols = activate[layer - 1][0].size();
                int curr_rows = activate[layer].size();
                int curr_cols = activate[layer][0].size();

                kernelTranspose.setArg(0, d_cweights[layer]);
                kernelTranspose.setArg(1, d_C_T_buf);
                kernelTranspose.setArg(2, cweights[layer].size());
                kernelTranspose.setArg(3, cweights[layer][0].size());
                cl::NDRange globalTranspose = calculate_global_2d(size2d, cweights[layer].size(), cweights[layer][0].size());
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, globalTranspose, local_2d));

                kernelMatMul.setArg(0, d_incoming[layer]);
                kernelMatMul.setArg(1, d_C_T_buf);
                kernelMatMul.setArg(2, d_grad_x_CT_buf);
                kernelMatMul.setArg(3, curr_rows);
                kernelMatMul.setArg(4, curr_cols);
                kernelMatMul.setArg(5, prev_cols);
                cl::NDRange globalMatMul = calculate_global_2d(size2d, curr_rows, prev_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, globalMatMul, local_2d));

                kernelDPow.setArg(0, d_activate[layer - 1]);
                kernelDPow.setArg(1, d_dprev_p_buf);
                kernelDPow.setArg(2, order);
                kernelDPow.setArg(3, (int)(prev_rows * prev_cols));
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelDPow, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_rows * prev_cols), local_1d));

                size_t prev_dot_size = dotProds[layer - 1].size() * dotProds[layer - 1][0].size();
                KernelReluDer.setArg(0, d_dotProds[layer - 1]);
                KernelReluDer.setArg(1, d_dprev_act_buf);
                KernelReluDer.setArg(2, (int)prev_dot_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(KernelReluDer, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_dot_size), local_1d));

                kernelHadamard2.setArg(0, d_grad_x_CT_buf);
                kernelHadamard2.setArg(1, d_dprev_p_buf);
                kernelHadamard2.setArg(2, d_dprev_act_buf);
                kernelHadamard2.setArg(3, d_incoming[layer - 1]);
                kernelHadamard2.setArg(4, prev_rows);
                kernelHadamard2.setArg(5, prev_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelHadamard2, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_rows * prev_cols), local_1d));

                kernelPower.setArg(0, d_activate[layer - 1]);
                kernelPower.setArg(1, d_dprev_p_buf); // Reuse d_dprev_p_buf as temp for prev_p
                kernelPower.setArg(2, order);
                kernelPower.setArg(3, (int)(prev_rows * prev_cols));
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelPower, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_rows * prev_cols), local_1d));

                kernelTranspose.setArg(0, d_dprev_p_buf);
                kernelTranspose.setArg(1, d_prev_p_T_buf);
                kernelTranspose.setArg(2, prev_rows);
                kernelTranspose.setArg(3, prev_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, calculate_global_2d(size2d, prev_rows, prev_cols), local_2d));

                kernelMatMul.setArg(0, d_prev_p_T_buf);
                kernelMatMul.setArg(1, d_incoming[layer]);
                kernelMatMul.setArg(2, d_gradC[layer]);
                kernelMatMul.setArg(3, prev_cols);
                kernelMatMul.setArg(4, prev_rows);
                kernelMatMul.setArg(5, curr_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, calculate_global_2d(size2d, prev_cols, curr_cols), local_2d));
                kernelScale.setArg(0, d_gradC[layer]);
                kernelScale.setArg(1, d_gradC[layer]);
                kernelScale.setArg(2, ALPHA);
                kernelScale.setArg(3, (int)(prev_cols * curr_cols));
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_cols * curr_cols), local_1d));

                kernelMatMul.setArg(0, d_ones_all_buf); // Use pre-filled ones buffer
                kernelMatMul.setArg(1, d_incoming[layer]);
                kernelMatMul.setArg(2, d_gradB[layer]);
                kernelMatMul.setArg(3, prev_cols);
                kernelMatMul.setArg(4, prev_rows);
                kernelMatMul.setArg(5, curr_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, calculate_global_2d(size2d, prev_cols, curr_cols), local_2d));
                kernelScale.setArg(0, d_gradB[layer]);
                kernelScale.setArg(1, d_gradB[layer]);
                kernelScale.setArg(2, 1.0f - ALPHA);
                kernelScale.setArg(3, (int)(prev_cols * curr_cols));
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_cols * curr_cols), local_1d));
            }

            // Backpropagation for the first layer
            int in_h = input.size();
            int in_w = input[0].size();
            int first_layer_cols = activate[0][0].size();

            kernelPower.setArg(0, d_in);
            kernelPower.setArg(1, d_dprev_p_buf); // Reuse temp buffer
            kernelPower.setArg(2, order);
            kernelPower.setArg(3, (int)(in_h * in_w));
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelPower, cl::NullRange, calculate_global_1d(WORKSIZE_1D, in_h * in_w), local_1d));

            kernelTranspose.setArg(0, d_dprev_p_buf);
            kernelTranspose.setArg(1, d_prev_p_T_buf);
            kernelTranspose.setArg(2, in_h);
            kernelTranspose.setArg(3, in_w);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, calculate_global_2d(size2d, in_h, in_w), local_2d));

            kernelMatMul.setArg(0, d_prev_p_T_buf);
            kernelMatMul.setArg(1, d_incoming[0]);
            kernelMatMul.setArg(2, d_gradC[0]);
            kernelMatMul.setArg(3, in_w);
            kernelMatMul.setArg(4, in_h);
            kernelMatMul.setArg(5, first_layer_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, calculate_global_2d(size2d, in_w, first_layer_cols), local_2d));
            kernelScale.setArg(0, d_gradC[0]);
            kernelScale.setArg(1, d_gradC[0]);
            kernelScale.setArg(2, ALPHA);
            kernelScale.setArg(3, (int)(in_w * first_layer_cols));
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, calculate_global_1d(WORKSIZE_1D, in_w * first_layer_cols), local_1d));

            kernelMatMul.setArg(0, d_ones_all_buf);
            kernelMatMul.setArg(1, d_incoming[0]);
            kernelMatMul.setArg(2, d_gradB[0]);
            kernelMatMul.setArg(3, in_w);
            kernelMatMul.setArg(4, in_h);
            kernelMatMul.setArg(5, first_layer_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, calculate_global_2d(size2d, in_w, first_layer_cols), local_2d));
            kernelScale.setArg(0, d_gradB[0]);
            kernelScale.setArg(1, d_gradB[0]);
            kernelScale.setArg(2, 1.0f - ALPHA);
            kernelScale.setArg(3, (int)(in_w * first_layer_cols));
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, calculate_global_1d(WORKSIZE_1D, in_w * first_layer_cols), local_1d));
            CL_CHECK(clCommandQueue.finish());

            // Update weights and copy to host vectors
            for (int i = 0; i < this->layers; ++i) {
                size_t c_size = cweights[i].size() * cweights[i][0].size();
                size_t b_size = bweights[i].size() * bweights[i][0].size();
                cl::NDRange globalUpdate = calculate_global_1d(WORKSIZE_1D, c_size);

                kernelUpdateWeights.setArg(0, d_cweights[i]);
                kernelUpdateWeights.setArg(1, d_gradC[i]);
                switch (weightUpdateType) {
                    case 0:
                        kernelUpdateWeights.setArg(2, learningRate);
                        kernelUpdateWeights.setArg(3, (int)c_size);
                        break;
                    case 1:
                        kernelUpdateWeights.setArg(2, (int)c_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, LAMBDA_L1);
                        break;
                    case 2:
                        kernelUpdateWeights.setArg(2, (int)c_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, LAMBDA_L2);
                        break;
                    case 3:
                        kernelUpdateWeights.setArg(2, (int)c_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, LAMBDA_L1);
                        kernelUpdateWeights.setArg(5, LAMBDA_L2);
                        break;
                    case 4:
                        kernelUpdateWeights.setArg(2, (int)c_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, WEIGHT_DECAY);
                        break;
                    case 5:
                        kernelUpdateWeights.setArg(2, (int)c_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, DROPOUT_RATE);
                        kernelUpdateWeights.setArg(5, (uint)rand());
                        break;
                }
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdateWeights, cl::NullRange, globalUpdate, local_1d));

                kernelUpdateWeights.setArg(0, d_bweights[i]);
                kernelUpdateWeights.setArg(1, d_gradB[i]);
                switch (weightUpdateType) {
                    case 0:
                        kernelUpdateWeights.setArg(2, learningRate);
                        kernelUpdateWeights.setArg(3, (int)b_size);
                        break;
                    case 1:
                        kernelUpdateWeights.setArg(2, (int)b_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, LAMBDA_L1);
                        break;
                    case 2:
                        kernelUpdateWeights.setArg(2, (int)b_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, LAMBDA_L2);
                        break;
                    case 3:
                        kernelUpdateWeights.setArg(2, (int)b_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, LAMBDA_L1);
                        kernelUpdateWeights.setArg(5, LAMBDA_L2);
                        break;
                    case 4:
                        kernelUpdateWeights.setArg(2, (int)b_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, WEIGHT_DECAY);
                        break;
                    case 5:
                        kernelUpdateWeights.setArg(2, (int)b_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, DROPOUT_RATE);
                        kernelUpdateWeights.setArg(5, (uint)rand());
                        break;
                }
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdateWeights, cl::NullRange, globalUpdate, local_1d));

                std::vector<float> c_upd(c_size), b_upd(b_size);
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_cweights[i], CL_TRUE, 0, sizeof(float) * c_size, (void*)c_upd.data()));
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_bweights[i], CL_TRUE, 0, sizeof(float) * b_size, (void*)b_upd.data()));
                cweights[i] = reshape(c_upd, cweights[i].size(), cweights[i][0].size());
                bweights[i] = reshape(b_upd, bweights[i].size(), bweights[i][0].size());
            }
        }

        this->learningRate = initialLR;
        if (epoch == maxEpochs) {
            std::cout << "Max epochs reached without convergence." << std::endl;
        }
        std::cout << "Training complete for this input-target pair." << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "Error during clBufTrain (mnn2d): " << e.what() << std::endl;
    }

    // --- Buffer Cleanup ---
    d_in = cl::Buffer(); d_exp = cl::Buffer(); d_out = cl::Buffer(); d_err = cl::Buffer();
    d_final_output = cl::Buffer(); d_C_T_buf = cl::Buffer(); d_prev_p_T_buf = cl::Buffer(); d_ones_all_buf = cl::Buffer(); d_partial_results_buf = cl::Buffer();
    d_grad_x_CT_buf = cl::Buffer(); d_dprev_p_buf = cl::Buffer(); d_dprev_act_buf = cl::Buffer();
    for (int i = 0; i < layers; ++i) {
        d_cweights[i] = cl::Buffer(); d_bweights[i] = cl::Buffer();
        d_gradC[i] = cl::Buffer(); d_gradB[i] = cl::Buffer();
        d_dotProds[i] = cl::Buffer(); d_activate[i] = cl::Buffer();
        d_incoming[i] = cl::Buffer();
        if (i < layers - 1) {
            d_dpow[i] = cl::Buffer(); d_dact[i] = cl::Buffer();
        }
    }
}

#endif // USE_CL