#ifdef USE_CL
#include "mnn.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm> // For std::max
#include <cmath>     // For std::ceil
#include <limits>    // For std::numeric_limits
#include <cstdlib>

/**
 * @brief trains the mnn network on a single input-target pair for 1 cycle using OpenCL.
 * @param input The input vector.
 * @param target The target output vector.
 * @param useBuffer 0 for stand alone functions or 1 for all-buffers-in-single function 
 */
void mnn::clTrain1c(const std::vector<float>& input, const std::vector<float>& target, bool useBuffer) {
    if (useBuffer == 0) {
        // 1. Forward propagation
        clForprop(input);

        if(maxIndex(output) == maxIndex(target)) {
            // std::cout << "Correct output predicted with loss " << crossEntropy(output, target) << "." << std::endl;
        }
        else {
            // check for error and break if acceptable
            currloss = crossEntropy(output, target);
            // std::cout << "Current CE Loss: " << currloss << std::endl;

            // 2. Backward propagation
            clBackprop(target);
            prevloss = currloss;
        }
    }
    else {
        // 1. Forward propagation
        cl_int err;
        cl::NDRange local_1d(WORKSIZE_1D);
        cl::NDRange local_2d(WORKSIZE_2DX, WORKSIZE_2DY);
        size_t size2d[2] = {WORKSIZE_2DX, WORKSIZE_2DY};

        // --- Buffer Allocation ---
        cl::Buffer d_in, d_exp, d_out, d_err;
        std::vector<cl::Buffer> d_cweights(layers), d_bweights(layers), d_gradC(layers), d_gradB(layers);
        std::vector<cl::Buffer> d_dotProds(layers), d_activate(layers), d_incoming(layers);
        cl::Buffer d_ones, d_preoutgoing_l, d_outgoing_l, d_dpow_l, d_dact_l;

        // Allocate input/output/target buffers
        d_in = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.size() * sizeof(float), (void*)input.data(), &err); CL_CHECK(err);
        d_exp = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, target.size() * sizeof(float), (void*)target.data(), &err); CL_CHECK(err);
        d_out = cl::Buffer(clContext, CL_MEM_READ_WRITE, output.size() * sizeof(float)); CL_CHECK(err);
        d_err = cl::Buffer(clContext, CL_MEM_READ_WRITE, output.size() * sizeof(float)); CL_CHECK(err);

        size_t max_layer_width = 0;
        for (int w : width) max_layer_width = std::max(max_layer_width, (size_t)w);
        max_layer_width = std::max(max_layer_width, input.size());
        max_layer_width = std::max(max_layer_width, output.size());
        std::vector<float> v1(max_layer_width, 1.0f);
        d_ones = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * max_layer_width, v1.data(), &err); CL_CHECK(err);

        // Allocate layer-specific buffers
        for (int i = 0; i < layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            size_t act_size = activate[i].size();
            d_cweights[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, c_size * sizeof(float)); CL_CHECK(err);
            d_bweights[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, b_size * sizeof(float)); CL_CHECK(err);
            d_gradC[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, c_size * sizeof(float)); CL_CHECK(err);
            d_gradB[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, b_size * sizeof(float)); CL_CHECK(err);
            d_dotProds[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
            d_activate[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
            d_incoming[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
        }

        if (layers > 1) {
            d_preoutgoing_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width); CL_CHECK(err);
            d_outgoing_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width); CL_CHECK(err);
            d_dpow_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width); CL_CHECK(err);
            d_dact_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width); CL_CHECK(err);
        }

        int epoch = 0;

        // Copy weights H2D for current iteration
        for (int i = 0; i < layers; ++i) {
            std::vector<float> flat_c = flatten(cweights[i]);
            std::vector<float> flat_b = flatten(bweights[i]);
            CL_CHECK(clCommandQueue.enqueueWriteBuffer(d_cweights[i], CL_TRUE, 0, flat_c.size() * sizeof(float), flat_c.data()));
            CL_CHECK(clCommandQueue.enqueueWriteBuffer(d_bweights[i], CL_TRUE, 0, flat_b.size() * sizeof(float), flat_b.data()));
        }

        // --- Forward Propagation (adapted from mnn::clForprop) ---
        cl::Buffer d_current_act = d_in;
        cl::Kernel kernelForward = kernels.at("kernelLayerForward2");
        cl::Kernel kernelSigmoid = kernels.at("sigmoid");

        // First layer
        int current_in_size = input.size();
        int current_out_size = width[0];
        kernelForward.setArg(0, d_current_act);
        kernelForward.setArg(1, d_dotProds[0]);
        kernelForward.setArg(2, d_cweights[0]);
        kernelForward.setArg(3, d_bweights[0]);
        kernelForward.setArg(4, current_in_size);
        kernelForward.setArg(5, current_out_size);
        kernelForward.setArg(6, order);
        cl::NDRange globalForward = calculate_global_1d(WORKSIZE_1D, current_out_size);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForward, cl::NullRange, globalForward, local_1d));

        kernelSigmoid.setArg(0, d_dotProds[0]);
        kernelSigmoid.setArg(1, d_activate[0]);
        kernelSigmoid.setArg(2, current_out_size);
        cl::NDRange globalSig = calculate_global_1d(WORKSIZE_1D, current_out_size);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSigmoid, cl::NullRange, globalSig, local_1d));

        // Subsequent layers
        for (int i = 1; i < layers; ++i) {
            d_current_act = d_activate[i - 1];
            current_in_size = width[i - 1];
            current_out_size = width[i];
            kernelForward.setArg(0, d_current_act);
            kernelForward.setArg(1, d_dotProds[i]);
            kernelForward.setArg(2, d_cweights[i]);
            kernelForward.setArg(3, d_bweights[i]);
            kernelForward.setArg(4, current_in_size);
            kernelForward.setArg(5, current_out_size);
            kernelForward.setArg(6, order);
            globalForward = calculate_global_1d(WORKSIZE_1D, current_out_size);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForward, cl::NullRange, globalForward, local_1d));

            kernelSigmoid.setArg(0, d_dotProds[i]);
            kernelSigmoid.setArg(1, d_activate[i]);
            kernelSigmoid.setArg(2, current_out_size);
            globalSig = calculate_global_1d(WORKSIZE_1D, current_out_size);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSigmoid, cl::NullRange, globalSig, local_1d));
        }
        CL_CHECK(clCommandQueue.finish());

        // Copy output D2H to check for correctness and loss
        CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[layers - 1], CL_TRUE, 0, output.size() * sizeof(float), output.data()));
        output = softmax(output);

        if(maxIndex(output) == maxIndex(target)) {
            float loss = crossEntropy(output, target);
            if (loss < 0) loss = 0;
            // std::cout << "Correct output predicted with loss " << loss << "." << std::endl;
        }
        else {
            zeroGradients();
            // check for error and break if acceptable
            currloss = crossEntropy(output, target);
            // std::cout << "Current CE Loss: " << currloss << std::endl;

            // 2. Backward propagation
            cl::Kernel kernelSub = kernels.at("subtract");
            cl::Kernel kernelSigmoidDer = kernels.at("sigmoidDer");
            cl::Kernel kernelScale = kernels.at("scaleByValue");
            cl::Kernel kernelDPow = kernels.at("dPower");
            cl::Kernel kernelVecxVec2Mat = kernels.at("vecxvec2mat");
            cl::Kernel kernelVecxMat2Vec = kernels.at("vecxmat2vec");
            cl::Kernel kernelTranspose = kernels.at("transpose");
            cl::Kernel kernelHadamard = kernels.at("hadamard2");
            
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

            // Calculate initial error (activate[layers - 1] - expected)
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[layers - 1], CL_TRUE, 0, output.size() * sizeof(float), output.data()));
            CL_CHECK(clCommandQueue.enqueueWriteBuffer(d_out, CL_TRUE, 0, output.size() * sizeof(float), output.data()));
            kernelSub.setArg(0, d_out);
            kernelSub.setArg(1, d_exp);
            kernelSub.setArg(2, d_err);
            kernelSub.setArg(3, (int)output.size());
            cl::NDRange globalSub = calculate_global_1d(WORKSIZE_1D, output.size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSub, cl::NullRange, globalSub, local_1d));
            CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_err, d_incoming[layers - 1], 0, 0, sizeof(float) * output.size()));

            size_t max_cweight_size = 0;
            for(int i = 0; i < layers; i++) max_cweight_size = std::max(max_cweight_size, cweights[i].size() * cweights[i][0].size());
            cl::Buffer d_C_T(clContext, CL_MEM_READ_WRITE, max_cweight_size * sizeof(float)); CL_CHECK(err);

            // Backpropagation loop (last layer to second layer)
            for (int layer = layers - 1; layer >= 1; --layer) {
                int prev_size = activate[layer - 1].size();
                int curr_size = activate[layer].size();
                int cweight_rows = prev_size;
                int cweight_cols = curr_size;
                size_t cweight_flat_size = cweight_rows * cweight_cols;

                cl::NDRange globalWeightGrad = calculate_global_1d(WORKSIZE_1D, cweight_flat_size);
                cl::NDRange globalOutGrad = calculate_global_1d(WORKSIZE_1D, prev_size);

                // dL/dC_l (Outer Product: d_activate[L-1] x d_incoming[L])
                kernelVecxVec2Mat.setArg(0, d_activate[layer - 1]);
                kernelVecxVec2Mat.setArg(1, d_incoming[layer]);
                kernelVecxVec2Mat.setArg(2, d_gradC[layer]);
                kernelVecxVec2Mat.setArg(3, cweight_rows);
                kernelVecxVec2Mat.setArg(4, cweight_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelVecxVec2Mat, cl::NullRange, globalWeightGrad, local_1d));

                // scale gradc by ALPHA
                kernelScale.setArg(0, d_gradC[layer]);
                kernelScale.setArg(1, d_gradC[layer]);
                kernelScale.setArg(2, ALPHA);
                kernelScale.setArg(3, (int)cweight_flat_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGrad, local_1d));

                // dL/dB_l (Outer Product: ones x d_incoming[L])
                kernelVecxVec2Mat.setArg(0, d_ones);
                kernelVecxVec2Mat.setArg(1, d_incoming[layer]);
                kernelVecxVec2Mat.setArg(2, d_gradB[layer]);
                kernelVecxVec2Mat.setArg(3, cweight_rows);
                kernelVecxVec2Mat.setArg(4, cweight_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelVecxVec2Mat, cl::NullRange, globalWeightGrad, local_1d));

                // scale gradb by 1-ALPHA
                kernelScale.setArg(0, d_gradB[layer]);
                kernelScale.setArg(1, d_gradB[layer]);
                kernelScale.setArg(2, 1.0f - ALPHA);
                kernelScale.setArg(3, (int)cweight_flat_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGrad, local_1d));

                // --- Outgoing Gradient Calculation (for layer-1) ---
                kernelTranspose.setArg(0, d_cweights[layer]);
                kernelTranspose.setArg(1, d_C_T);
                kernelTranspose.setArg(2, cweight_rows);
                kernelTranspose.setArg(3, cweight_cols);
                cl::NDRange globalTranspose = calculate_global_2d(size2d, cweight_rows, cweight_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, globalTranspose, local_2d));

                // incoming gradient x C^T
                kernelVecxMat2Vec.setArg(0, d_incoming[layer]);
                kernelVecxMat2Vec.setArg(1, d_C_T);
                kernelVecxMat2Vec.setArg(2, d_preoutgoing_l);
                kernelVecxMat2Vec.setArg(3, cweight_cols); // matRows
                kernelVecxMat2Vec.setArg(4, cweight_rows); // matCols
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelVecxMat2Vec, cl::NullRange, globalOutGrad, local_1d));

                // derivative of power
                kernelDPow.setArg(0, d_activate[layer - 1]);
                kernelDPow.setArg(1, d_dpow_l);
                kernelDPow.setArg(2, order);
                kernelDPow.setArg(3, prev_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelDPow, cl::NullRange, globalOutGrad, local_1d));

                // derivative of activation
                kernelSigmoidDer.setArg(0, d_dotProds[layer - 1]);
                kernelSigmoidDer.setArg(1, d_dact_l);
                kernelSigmoidDer.setArg(2, prev_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSigmoidDer, cl::NullRange, globalOutGrad, local_1d));

                // outgoing gradient = (dl/dz_l x C^T) . dprev_p . dprevAct
                kernelHadamard.setArg(0, d_preoutgoing_l);
                kernelHadamard.setArg(1, d_dpow_l);
                kernelHadamard.setArg(2, d_dact_l);
                kernelHadamard.setArg(3, d_incoming[layer - 1]);
                kernelHadamard.setArg(4, 1);
                kernelHadamard.setArg(5, prev_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelHadamard, cl::NullRange, globalOutGrad, local_1d));
            }

            // Backpropagation for the first layer (input layer)
            int prev_size = input.size();
            int curr_size = activate[0].size();
            int cweight_rows = prev_size;
            int cweight_cols = curr_size;
            size_t cweight_flat_size = cweight_rows * cweight_cols;
            cl::NDRange globalWeightGradFirst = calculate_global_1d(WORKSIZE_1D, cweight_flat_size);

            // dL/dC_0 (Outer Product: d_in x d_incoming[0])
            kernelVecxVec2Mat.setArg(0, d_in);
            kernelVecxVec2Mat.setArg(1, d_incoming[0]);
            kernelVecxVec2Mat.setArg(2, d_gradC[0]);
            kernelVecxVec2Mat.setArg(3, cweight_rows);
            kernelVecxVec2Mat.setArg(4, cweight_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelVecxVec2Mat, cl::NullRange, globalWeightGradFirst, local_1d));

            // scale gradc by ALPHA
            kernelScale.setArg(0, d_gradC[0]);
            kernelScale.setArg(1, d_gradC[0]);
            kernelScale.setArg(2, ALPHA);
            kernelScale.setArg(3, (int)cweight_flat_size);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGradFirst, local_1d));

            // dL/dB_0 (Outer Product: ones x d_incoming[0])
            kernelVecxVec2Mat.setArg(0, d_ones);
            kernelVecxVec2Mat.setArg(1, d_incoming[0]);
            kernelVecxVec2Mat.setArg(2, d_gradB[0]);
            kernelVecxVec2Mat.setArg(3, cweight_rows);
            kernelVecxVec2Mat.setArg(4, cweight_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelVecxVec2Mat, cl::NullRange, globalWeightGradFirst, local_1d));

            // scale gradb by 1-ALPHA
            kernelScale.setArg(0, d_gradB[0]);
            kernelScale.setArg(1, d_gradB[0]);
            kernelScale.setArg(2, 1.0f - ALPHA);
            kernelScale.setArg(3, (int)cweight_flat_size);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGradFirst, local_1d));
            CL_CHECK(clCommandQueue.finish());

            // Update weights and copy to host vectors
            for (int i = 0; i < this->layers; ++i) {
                size_t c_size = cweights[i].size() * cweights[i][0].size();
                size_t b_size = bweights[i].size() * bweights[i][0].size();
                cl::NDRange globalUpdate = calculate_global_1d(WORKSIZE_1D, c_size);

                // Update C weights using kernelUpdateWeights
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
                // Update B weights using kernelUpdateWeights
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

                // copy and reshape
                std::vector<float> c_upd(c_size), b_upd(b_size), c_grad(c_size), b_grad(b_size);
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[i], CL_TRUE, 0, sizeof(float) * activate[i].size(), (void*)activate[i].data()));
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_cweights[i], CL_TRUE, 0, sizeof(float) * c_size, (void*)c_upd.data()));
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_bweights[i], CL_TRUE, 0, sizeof(float) * b_size, (void*)b_upd.data()));
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_gradC[i], CL_TRUE, 0, sizeof(float) * c_size, (void*)c_grad.data()));
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_gradB[i], CL_TRUE, 0, sizeof(float) * b_size, (void*)b_grad.data()));

                cweights[i] = reshape(c_upd, cweights[i].size(), cweights[i][0].size());
                bweights[i] = reshape(b_upd, bweights[i].size(), bweights[i][0].size());
                cgradients[i] = reshape(c_grad, cgradients[i].size(), cgradients[i][0].size());
                bgradients[i] = reshape(b_grad, bgradients[i].size(), bgradients[i][0].size());
            }
        }
        // release all buffers
        for (int i = 0; i < layers; ++i) {
            d_cweights[i] = cl::Buffer();
            d_bweights[i] = cl::Buffer();
            d_gradC[i] = cl::Buffer();
            d_gradB[i] = cl::Buffer();
            d_dotProds[i] = cl::Buffer();
            d_activate[i] = cl::Buffer();
            d_incoming[i] = cl::Buffer();
        }
    }
}


/**
 * @brief trains the mnn2d network on a single input-target pair using OpenCL.
 * @param input The input matrix.
 * @param target The target vector (corresponding to the pooled output).
 * @param useBuffer 0 for stand alone functions or 1 for all-buffers-in-single function 
 */
void mnn2d::clTrain1c(const std::vector<std::vector<float>>& input, const std::vector<float>& target, bool useBuffer) {
    if (useBuffer == 0) {
        // 1. Forward propagation
        clForprop(input);

        if(maxIndex(output) == maxIndex(target)) {
            // std::cout << "Correct output predicted with loss " << crossEntropy(output, target) << "." << std::endl;
        }
        else {
            // check for error and break if acceptable
            currloss = crossEntropy(output, target);
            // std::cout << "Current CE Loss: " << currloss << std::endl;

            // 2. Backward propagation
            clBackprop(target);
            prevloss = currloss;
        }
    }
    else {
        // 1. Forward propagation
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
        d_dprev_act_buf = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_act_size); CL_CHECK(err);

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

        for (int i = 0; i < layers; ++i) {
            std::vector<float> flat_c = flatten(cweights[i]);
            std::vector<float> flat_b = flatten(bweights[i]);
            CL_CHECK(clCommandQueue.enqueueWriteBuffer(d_cweights[i], CL_TRUE, 0, flat_c.size() * sizeof(float), flat_c.data()));
            CL_CHECK(clCommandQueue.enqueueWriteBuffer(d_bweights[i], CL_TRUE, 0, flat_b.size() * sizeof(float), flat_b.data()));
        }

        // --- Forward Propagation (adapted from mnn2d::clForprop) ---
        cl::Buffer d_current_act = d_in;
        cl::Kernel kernelForward = kernels.at("kernelLayerForward4");
        cl::Kernel kernelRelu = kernels.at("relu");
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
        kernelRelu.setArg(0, d_dotProds[0]);
        kernelRelu.setArg(1, d_activate[0]);
        kernelRelu.setArg(2, (int)dotprod_size_layer0);
        cl::NDRange globalRelu = calculate_global_1d(WORKSIZE_1D, dotprod_size_layer0);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelRelu, cl::NullRange, globalRelu, local_1d));

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
            kernelRelu.setArg(0, d_dotProds[i]);
            kernelRelu.setArg(1, d_activate[i]);
            kernelRelu.setArg(2, (int)dotprod_size_layer_i);
            globalRelu = calculate_global_1d(WORKSIZE_1D, dotprod_size_layer_i);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelRelu, cl::NullRange, globalRelu, local_1d));
        }

        // Mean pool the final activation layer
        cl::Buffer d_final_output(clContext, CL_MEM_WRITE_ONLY, sizeof(float) * outSize); CL_CHECK(err);
        kernelMeanPool.setArg(0, d_activate[layers - 1]);
        kernelMeanPool.setArg(1, d_final_output);
        kernelMeanPool.setArg(2, inHeight);
        kernelMeanPool.setArg(3, outSize);
        kernelMeanPool.setArg(4, 1); // poolSize
        cl::NDRange globalPool = calculate_global_1d(WORKSIZE_1D, outSize);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMeanPool, cl::NullRange, globalPool, local_1d));
        CL_CHECK(clCommandQueue.finish());

        // Copy output D2H to check for correctness and loss
        CL_CHECK(clCommandQueue.enqueueReadBuffer(d_final_output, CL_TRUE, 0, output.size() * sizeof(float), output.data()));
        output = softmax(output);

        if(maxIndex(output) == maxIndex(target)) {
            float loss = crossEntropy(output, target);
            if (loss < 0) loss = 0;
            // std::cout << "Correct output predicted with loss " << loss << "." << std::endl;
        }
        else {
            // check for error and break if acceptable
            currloss = crossEntropy(output, target);
            // std::cout << "Current CE Loss: " << currloss << std::endl;

            // 2. Backward propagation
            cl::Kernel kernelSub = kernels.at("subtract");
            cl::Kernel kernelReluDer = kernels.at("reluDer");
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

            // Initial error (pool(activate[layers - 1]) - expected)
            kernelSub.setArg(0, d_final_output);
            kernelSub.setArg(1, d_exp);
            kernelSub.setArg(2, d_err);
            kernelSub.setArg(3, (int)output.size());
            cl::NDRange globalSub = calculate_global_1d(WORKSIZE_1D, output.size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSub, cl::NullRange, globalSub, local_1d));

            // Distribute the error from d_err to each row of the last layer's incoming gradient buffer.
            for (size_t r = 0; r < activate[layers - 1].size(); ++r) {
                CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_err, d_incoming[layers - 1], 0, r * output.size() * sizeof(float), sizeof(float) * output.size()));
            }

            // Scale the incoming gradient by 1/inHeight (Derivative of Mean Pool)
            kernelScale.setArg(0, d_incoming[layers - 1]);
            kernelScale.setArg(1, d_incoming[layers - 1]);
            kernelScale.setArg(2, 1.0f / (float)activate[layers - 1].size());
            kernelScale.setArg(3, (int)(activate[layers - 1].size() * output.size()));
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, calculate_global_1d(WORKSIZE_1D, activate[layers - 1].size() * output.size()), local_1d));

            // Backpropagation from last to second layer
            for (int layer = layers - 1; layer >= 1; --layer) {
                int prev_rows = activate[layer - 1].size();
                int prev_cols = activate[layer - 1][0].size();
                int curr_rows = activate[layer].size();
                int curr_cols = activate[layer][0].size();

                cl::Buffer d_C_T(clContext, CL_MEM_READ_WRITE, cweights[layer].size() * cweights[layer][0].size() * sizeof(float)); CL_CHECK(err);
                kernelTranspose.setArg(0, d_cweights[layer]);
                kernelTranspose.setArg(1, d_C_T);
                kernelTranspose.setArg(2, prev_cols);
                kernelTranspose.setArg(3, curr_cols);
                cl::NDRange globalTranspose = calculate_global_2d(size2d, prev_cols, curr_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, globalTranspose, local_2d));

                kernelMatMul.setArg(0, d_incoming[layer]);
                kernelMatMul.setArg(1, d_C_T);
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
                kernelReluDer.setArg(0, d_dotProds[layer - 1]);
                kernelReluDer.setArg(1, d_dprev_act_buf);
                kernelReluDer.setArg(2, (int)prev_dot_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelReluDer, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_dot_size), local_1d));

                kernelHadamard2.setArg(0, d_grad_x_CT_buf);
                kernelHadamard2.setArg(1, d_dprev_p_buf);
                kernelHadamard2.setArg(2, d_dprev_act_buf);
                kernelHadamard2.setArg(3, d_incoming[layer - 1]);
                kernelHadamard2.setArg(4, prev_rows);
                kernelHadamard2.setArg(5, prev_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelHadamard2, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_rows * prev_cols), local_1d));

                cl::Buffer d_prev_p(clContext, CL_MEM_READ_WRITE, prev_rows * prev_cols * sizeof(float)); CL_CHECK(err);
                kernelPower.setArg(0, d_activate[layer - 1]);
                kernelPower.setArg(1, d_prev_p);
                kernelPower.setArg(2, order);
                kernelPower.setArg(3, (int)(prev_rows * prev_cols));
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelPower, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_rows * prev_cols), local_1d));

                cl::Buffer d_prev_p_T(clContext, CL_MEM_READ_WRITE, prev_cols * prev_rows * sizeof(float)); CL_CHECK(err);
                kernelTranspose.setArg(0, d_prev_p);
                kernelTranspose.setArg(1, d_prev_p_T);
                kernelTranspose.setArg(2, prev_rows);
                kernelTranspose.setArg(3, prev_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, calculate_global_2d(size2d, prev_rows, prev_cols), local_2d));

                kernelMatMul.setArg(0, d_prev_p_T);
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

                std::vector<float> ones_vec(prev_cols * prev_rows, 1.0f);
                cl::Buffer d_onesT(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ones_vec.size(), ones_vec.data(), &err); CL_CHECK(err);
                kernelMatMul.setArg(0, d_onesT);
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

                d_C_T = cl::Buffer(); d_prev_p = cl::Buffer(); d_prev_p_T = cl::Buffer(); d_onesT = cl::Buffer();
            }

            // Backpropagation for the first layer
            int in_h = input.size();
            int in_w = input[0].size();
            int first_layer_cols = activate[0][0].size();

            cl::Buffer d_input_p(clContext, CL_MEM_READ_WRITE, in_h * in_w * sizeof(float)); CL_CHECK(err);
            kernelPower.setArg(0, d_in);
            kernelPower.setArg(1, d_input_p);
            kernelPower.setArg(2, order);
            kernelPower.setArg(3, (int)(in_h * in_w));
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelPower, cl::NullRange, calculate_global_1d(WORKSIZE_1D, in_h * in_w), local_1d));

            cl::Buffer d_input_p_T(clContext, CL_MEM_READ_WRITE, in_w * in_h * sizeof(float)); CL_CHECK(err);
            kernelTranspose.setArg(0, d_input_p);
            kernelTranspose.setArg(1, d_input_p_T);
            kernelTranspose.setArg(2, in_h);
            kernelTranspose.setArg(3, in_w);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, calculate_global_2d(size2d, in_h, in_w), local_2d));

            kernelMatMul.setArg(0, d_input_p_T);
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

            std::vector<float> ones_vec(in_w * in_h, 1.0f);
            cl::Buffer d_ones_T(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ones_vec.size(), ones_vec.data(), &err); CL_CHECK(err);
            kernelMatMul.setArg(0, d_ones_T);
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

            d_input_p = cl::Buffer(); d_input_p_T = cl::Buffer(); d_ones_T = cl::Buffer();

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

                std::vector<float> c_upd(c_size), b_upd(b_size), c_grad(c_size), b_grad(b_size), acti(activate[i].size() * activate[i][0].size());
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[i], CL_TRUE, 0, sizeof(float) * acti.size(), (void*)acti.data()));
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_cweights[i], CL_TRUE, 0, sizeof(float) * c_size, (void*)c_upd.data()));
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_bweights[i], CL_TRUE, 0, sizeof(float) * b_size, (void*)b_upd.data()));
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_gradC[i], CL_TRUE, 0, sizeof(float) * c_size, (void*)c_grad.data()));
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_gradB[i], CL_TRUE, 0, sizeof(float) * b_size, (void*)b_grad.data()));

                activate[i] = reshape(acti, activate[i].size(), activate[i][0].size());
                cweights[i] = reshape(c_upd, cweights[i].size(), cweights[i][0].size());
                bweights[i] = reshape(b_upd, bweights[i].size(), bweights[i][0].size());
                cgradients[i] = reshape(c_grad, cgradients[i].size(), cgradients[i][0].size());
                bgradients[i] = reshape(b_grad, bgradients[i].size(), bgradients[i][0].size());
            }
        }
        // release the buffers
        for (int i = 0; i < layers; ++i) {
            d_cweights[i] = cl::Buffer();
            d_bweights[i] = cl::Buffer();
            d_gradC[i] = cl::Buffer();
            d_gradB[i] = cl::Buffer();
            d_dotProds[i] = cl::Buffer();
            d_activate[i] = cl::Buffer();
        }
    }
}

#endif // USE_CU