#ifdef USE_CL
#include "mnn1d.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <algorithm> // For std::copy
#include <cstdlib>   // For rand()
#include <cmath> // For std::ceil
#include <iostream>

/**
 * @brief Backpropagation for mnn using OpenCL.
 * @param expected The expected output vector.
 */
void mnn1d::clBackprop(const std::vector<float>& expected) {
    try {
        cl_int err;
        cl::NDRange local_1d(WORKSIZE_1D);
        cl::NDRange local_2d(WORKSIZE_2DX, WORKSIZE_2DY);
        size_t size2d[2] = {WORKSIZE_2DX, WORKSIZE_2DY};

        size_t inputSize = input.size();
        size_t outputSize = output.size();

        cl::Buffer d_in, d_exp, d_out, d_err;
        std::vector<cl::Buffer> d_incoming(this->layers);
        std::vector<cl::Buffer> d_cweights(this->layers);
        std::vector<cl::Buffer> d_bweights(this->layers);
        std::vector<cl::Buffer> d_gradC(this->layers);
        std::vector<cl::Buffer> d_gradB(this->layers);
        std::vector<cl::Buffer> d_dotProds(this->layers);
        std::vector<cl::Buffer> d_activate(this->layers);
        cl::Buffer d_ones, d_preoutgoing_l, d_outgoing_l, d_dpow_l, d_dact_l;

        auto kernelSigmoidDer = kernels.at("sigmoidDer");
        auto kernelSub = kernels.at("subtract");
        auto kernelScale = kernels.at("scaleByValue");
        auto kernelDPow = kernels.at("dPower");
        auto kernelVecxVec2Mat = kernels.at("vecxvec2mat");
        auto kernelVecxMat2Vec = kernels.at("vecxmat2vec");
        auto kernelTranspose = kernels.at("transpose");
        auto kernelHadamard = kernels.at("hadamard2");

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

        // load values to buffers
        d_in = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputSize, input.data(), &err); CL_CHECK(err);
        d_exp = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * expected.size(), (void*)expected.data(), &err); CL_CHECK(err);
        d_out = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * output.size(), (void*)output.data(), &err); CL_CHECK(err);
        d_err = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * output.size()); CL_CHECK(err);

        for(int i = 0; i < this->layers; i++) {
            // allot weights to c and b and their gradients
            size_t cweight_size = cweights[i].size() * cweights[i][0].size();
            size_t bweight_size = bweights[i].size() * bweights[i][0].size();
            d_cweights[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * cweight_size, (void*)flatten(cweights[i]).data(), &err); CL_CHECK(err);
            d_bweights[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * bweight_size, (void*)flatten(bweights[i]).data(), &err); CL_CHECK(err);
            // fill d_act and d_dotProds
            d_activate[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * activate[i].size(), (void*)activate[i].data(), &err); CL_CHECK(err);
            d_dotProds[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * dotProds[i].size(), (void*)dotProds[i].data(), &err); CL_CHECK(err);
            // allot space to gradients
            d_gradC[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * cweights[i].size() * cweights[i][0].size()); CL_CHECK(err);
            d_gradB[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * bweights[i].size() * bweights[i][0].size()); CL_CHECK(err);
            // for incoming gradients
            d_incoming[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * activate[i].size()); CL_CHECK(err);
        }

        size_t max_layer_width = 0;
        for (const auto& w : width) { max_layer_width = std::max(max_layer_width, (size_t)w); }
        max_layer_width = std::max(max_layer_width, outputSize);
        max_layer_width = std::max(max_layer_width, inputSize);

        d_preoutgoing_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width);
        d_outgoing_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width);
        d_dpow_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width);
        d_dact_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width);

        std::vector<float> v1(max_layer_width, 1.0f);
        d_ones = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * v1.size(), v1.data(), &err); CL_CHECK(err);

        // Calculate initial error (output - expected)
        kernelSub.setArg(0, d_out);
        kernelSub.setArg(1, d_exp);
        kernelSub.setArg(2, d_err);
        kernelSub.setArg(3, (int)outputSize);
        cl::NDRange globalSub = calculate_global_1d(WORKSIZE_1D, outputSize);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSub, cl::NullRange, globalSub, local_1d));
        CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_err, d_incoming[layers - 1], 0, 0, sizeof(float) * outputSize));

        // Backpropagation loop (last layer to second layer)
        for(int layer = layers - 1; layer >= 1; layer--) {
            int prev_size = activate[layer - 1].size();
            int curr_size = activate[layer].size();
            int cweight_rows = prev_size;
            int cweight_cols = curr_size;
            size_t cweight_size = cweights[layer].size() * cweights[layer][0].size();
            size_t bweight_size = bweights[layer].size() * bweights[layer][0].size();

            cl::NDRange globalWeightGrad = calculate_global_1d(WORKSIZE_1D, cweight_size);
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
            kernelScale.setArg(3, (int)cweight_size);
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
            kernelScale.setArg(3, (int)bweight_size);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGrad, local_1d));

            // --- Outgoing Gradient Calculation (for layer-1) ---
            cl::Buffer d_C_T(clContext, CL_MEM_READ_WRITE, cweight_size * sizeof(float));
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
        size_t cweight_size = cweight_rows * cweight_cols;
        size_t bweight_size = cweight_rows * cweight_cols;
        cl::NDRange globalWeightGradFirst = calculate_global_1d(WORKSIZE_1D, cweight_size);

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
        kernelScale.setArg(3, (int)cweight_size);
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
        kernelScale.setArg(3, (int)bweight_size);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGradFirst, local_1d));

        // update weights and copy to host vectors
        for (int i = 0; i < this->layers; ++i) {
            size_t cweight_size = cweights[i].size() * cweights[i][0].size();
            size_t bweight_size = bweights[i].size() * bweights[i][0].size();
            std::vector<float> takeIn(cweight_size, 0.0f);
            std::vector<float> takeIn2(bweight_size, 0.0f);
            std::vector<float> takeIn3(cweight_size, 0.0f);
            std::vector<float> takeIn4(bweight_size, 0.0f);
            cl::NDRange globalWeightGrad = calculate_global_1d(WORKSIZE_1D, cweight_size);

            // Update C weights using kernelUpdateWeights
            kernelUpdateWeights.setArg(0, d_cweights[i]);
            kernelUpdateWeights.setArg(1, d_gradC[i]);
            switch (weightUpdateType) {
                case 0:
                    kernelUpdateWeights.setArg(2, learningRate);
                    kernelUpdateWeights.setArg(3, (int)cweight_size);
                    break;
                case 1:
                    kernelUpdateWeights.setArg(2, (int)cweight_size);
                    kernelUpdateWeights.setArg(3, learningRate);
                    kernelUpdateWeights.setArg(4, LAMBDA_L1);
                    break;
                case 2:
                    kernelUpdateWeights.setArg(2, (int)cweight_size);
                    kernelUpdateWeights.setArg(3, learningRate);
                    kernelUpdateWeights.setArg(4, LAMBDA_L2);
                    break;
                case 3:
                    kernelUpdateWeights.setArg(2, (int)cweight_size);
                    kernelUpdateWeights.setArg(3, learningRate);
                    kernelUpdateWeights.setArg(4, LAMBDA_L1);
                    kernelUpdateWeights.setArg(5, LAMBDA_L2);
                    break;
                case 4:
                    kernelUpdateWeights.setArg(2, (int)cweight_size);
                    kernelUpdateWeights.setArg(3, learningRate);
                    kernelUpdateWeights.setArg(4, WEIGHT_DECAY);
                    break;
                case 5:
                    kernelUpdateWeights.setArg(2, (int)cweight_size);
                    kernelUpdateWeights.setArg(3, learningRate);
                    kernelUpdateWeights.setArg(4, DROPOUT_RATE);
                    kernelUpdateWeights.setArg(5, (uint)rand());
                    break;
            }
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdateWeights, cl::NullRange, globalWeightGrad, local_1d));
            // Update B weights using kernelUpdateWeights
            kernelUpdateWeights.setArg(0, d_bweights[i]);
            kernelUpdateWeights.setArg(1, d_gradB[i]);
            switch (weightUpdateType) {
                case 0:
                    kernelUpdateWeights.setArg(2, learningRate);
                    kernelUpdateWeights.setArg(3, (int)bweight_size);
                    break;
                case 1:
                    kernelUpdateWeights.setArg(2, (int)bweight_size);
                    kernelUpdateWeights.setArg(3, learningRate);
                    kernelUpdateWeights.setArg(4, LAMBDA_L1);
                    break;
                case 2:
                    kernelUpdateWeights.setArg(2, (int)bweight_size);
                    kernelUpdateWeights.setArg(3, learningRate);
                    kernelUpdateWeights.setArg(4, LAMBDA_L2);
                    break;
                case 3:
                    kernelUpdateWeights.setArg(2, (int)bweight_size);
                    kernelUpdateWeights.setArg(3, learningRate);
                    kernelUpdateWeights.setArg(4, LAMBDA_L1);
                    kernelUpdateWeights.setArg(5, LAMBDA_L2);
                    break;
                case 4:
                    kernelUpdateWeights.setArg(2, (int)bweight_size);
                    kernelUpdateWeights.setArg(3, learningRate);
                    kernelUpdateWeights.setArg(4, WEIGHT_DECAY);
                    break;
                case 5:
                    kernelUpdateWeights.setArg(2, (int)bweight_size);
                    kernelUpdateWeights.setArg(3, learningRate);
                    kernelUpdateWeights.setArg(4, DROPOUT_RATE);
                    kernelUpdateWeights.setArg(5, (uint)rand());
                    break;
            }
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdateWeights, cl::NullRange, globalWeightGrad, local_1d));

            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[i], CL_TRUE, 0, sizeof(float) * activate[i].size(), (void*)activate[i].data()));
            // copy and reshape
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_cweights[i], CL_TRUE, 0, sizeof(float) * cweight_size, (void*)takeIn.data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_bweights[i], CL_TRUE, 0, sizeof(float) * bweight_size, (void*)takeIn2.data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_gradC[i], CL_TRUE, 0, sizeof(float) * cweight_size, (void*)takeIn3.data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_gradB[i], CL_TRUE, 0, sizeof(float) * bweight_size, (void*)takeIn4.data()));
            cweights[i] = reshape(takeIn, cweights[i].size(), cweights[i][0].size());
            bweights[i] = reshape(takeIn2, bweights[i].size(), bweights[i][0].size());
            cgradients[i] = reshape(takeIn3, cgradients[i].size(), cgradients[i][0].size());
            bgradients[i] = reshape(takeIn4, bgradients[i].size(), bgradients[i][0].size());

        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn1d::clBackprop: ") + e.what());
    }
}

#endif