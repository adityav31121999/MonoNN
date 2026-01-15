#ifdef USE_CL
#include "mnn.hpp"
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
void mnn::clBackprop(const std::vector<float>& expected) {
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
        throw std::runtime_error(std::string("Exception in mnn::clBackprop: ") + e.what());
    }
}


/**
 * @brief Backpropagation for mnn2d using OpenCL.
 * @param expected The expected output vector.
 */
void mnn2d::clBackprop(const std::vector<float>& expected) {
    try {
        cl_int err;
        cl::NDRange local_1d(WORKSIZE_1D);
        cl::NDRange local_2d(WORKSIZE_2DX, WORKSIZE_2DY);
        size_t size2d[2] = {WORKSIZE_2DX, WORKSIZE_2DY};
        // --- Buffer Allocation and Initialization ---
        cl::Buffer d_in, d_exp, d_out, d_err;
        std::vector<cl::Buffer> d_incoming(this->layers);
        std::vector<cl::Buffer> d_cweights(this->layers);
        std::vector<cl::Buffer> d_bweights(this->layers);
        std::vector<cl::Buffer> d_gradC(this->layers);
        std::vector<cl::Buffer> d_gradB(this->layers);
        std::vector<cl::Buffer> d_dotProds(this->layers);
        std::vector<cl::Buffer> d_activate(this->layers);
        std::vector<cl::Buffer> d_dpow(this->layers-1);
        std::vector<cl::Buffer> d_dact(this->layers-1);
        std::vector<cl::Buffer> d_preoutgoing(this->layers-1);
        std::vector<cl::Buffer> d_outgoing(this->layers-1);

        d_in = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size() * input[0].size(), flatten(input).data(), &err); CL_CHECK(err);
        d_out = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * output.size(), output.data(), &err); CL_CHECK(err);
        d_exp = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * expected.size(), (void*)expected.data(), &err); CL_CHECK(err);
        d_err = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * expected.size()); CL_CHECK(err);

        // Kernels
        auto kernelSub = kernels.at("subtract");
        auto kernelReluDer = kernels.at("reluDer");
        auto kernelHadamard2 = kernels.at("hadamard2");
        auto kernelMatMul = kernels.at("matxmat2mat");
        auto kernelScale = kernels.at("scaleByValue");
        auto kernelTranspose = kernels.at("transpose");
        auto kernelDPow = kernels.at("dPower");
        auto kernelPower = kernels.at("power");
        auto kernelMeanPool = kernels.at("meanPool");

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

        for (int i = 0; i < this->layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            size_t dot_size = dotProds[i].size() * dotProds[i][0].size();
            size_t act_size = activate[i].size() * activate[i][0].size();

            // weights
            d_cweights[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, c_size * sizeof(float), (void*)flatten(cweights[i]).data(), &err); CL_CHECK(err);
            d_bweights[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, b_size * sizeof(float), (void*)flatten(bweights[i]).data(), &err); CL_CHECK(err);
            // matrix products and activations
            d_dotProds[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dot_size * sizeof(float), (void*)flatten(dotProds[i]).data(), &err); CL_CHECK(err);
            d_activate[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, act_size * sizeof(float), (void*)flatten(activate[i]).data(), &err); CL_CHECK(err);
            // gradients
            d_gradC[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, c_size * sizeof(float)); CL_CHECK(err);
            d_gradB[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, b_size * sizeof(float)); CL_CHECK(err);
            d_incoming[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
        }
        for (int i = 0; i < this->layers - 1; ++i) {
            size_t act_size = activate[i].size() * activate[i][0].size();
            d_outgoing[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
            d_dpow[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
            d_dact[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
            d_preoutgoing[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
        }


        int last_layer_rows = inHeight, last_layer_cols = outSize;
        d_out = cl::Buffer(clContext, CL_MEM_WRITE_ONLY, sizeof(float) * last_layer_cols);
        kernelMeanPool.setArg(0, d_activate[layers - 1]);
        kernelMeanPool.setArg(1, d_out);
        kernelMeanPool.setArg(2, last_layer_rows);
        kernelMeanPool.setArg(3, last_layer_cols);
        kernelMeanPool.setArg(4, 1);
        cl::NDRange globalPool = calculate_global_1d(WORKSIZE_1D, last_layer_cols);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMeanPool, cl::NullRange, globalPool, local_1d));

        // --- Backpropagation ---
        // Initial error (output - expected)
        kernelSub.setArg(0, d_out);
        kernelSub.setArg(1, d_exp);
        kernelSub.setArg(2, d_err);
        kernelSub.setArg(3, (int)output.size());
        cl::NDRange globalSub = calculate_global_1d(WORKSIZE_1D, output.size());
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSub, cl::NullRange, globalSub, local_1d));

        // Distribute the error from d_err to each row of the last layer's incoming gradient buffer.
        for(size_t i = 0; i < activate[layers-1].size(); ++i) {
            CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_err, d_incoming[layers-1], 0, i * output.size() * sizeof(float), sizeof(float) * output.size()));
        }

        // Scale the incoming gradient by 1/inHeight (Derivative of Mean Pool)
        kernelScale.setArg(0, d_incoming[layers - 1]);
        kernelScale.setArg(1, d_incoming[layers - 1]);
        kernelScale.setArg(2, 1.0f / (float)activate[layers - 1].size());
        kernelScale.setArg(3, (int)(activate[layers - 1].size() * output.size()));
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, calculate_global_1d(WORKSIZE_1D, activate[layers - 1].size() * output.size()), local_1d));

        // Backpropagation from last to second layer
        for (int layer = layers - 1; layer >= 1; --layer) {
            int prev_rows = activate[layer-1].size();
            int prev_cols = activate[layer-1][0].size();
            int curr_rows = activate[layer].size();
            int curr_cols = activate[layer][0].size();

            // transpose
            cl::Buffer d_C_T(clContext, CL_MEM_READ_WRITE, curr_cols * prev_cols * sizeof(float));
            kernelTranspose.setArg(0, d_cweights[layer]);
            kernelTranspose.setArg(1, d_C_T);
            kernelTranspose.setArg(2, prev_cols);
            kernelTranspose.setArg(3, curr_cols);
            cl::NDRange globalTranspose = calculate_global_2d(size2d, prev_cols, curr_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, globalTranspose, local_2d));

            // dL/dz_l x C^T
            cl::Buffer d_grad_x_CT(clContext, CL_MEM_READ_WRITE, prev_rows * prev_cols * sizeof(float));
            kernelMatMul.setArg(0, d_incoming[layer]);
            kernelMatMul.setArg(1, d_C_T);
            kernelMatMul.setArg(2, d_grad_x_CT);
            kernelMatMul.setArg(3, curr_rows);
            kernelMatMul.setArg(4, curr_cols);
            kernelMatMul.setArg(5, prev_cols);
            cl::NDRange globalMatMul = calculate_global_2d(size2d, curr_rows, prev_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, globalMatMul, local_2d));

            //  d(prev_p)
            cl::Buffer d_dprev_p(clContext, CL_MEM_READ_WRITE, prev_rows * prev_cols * sizeof(float));
            kernelDPow.setArg(0, d_activate[layer-1]);
            kernelDPow.setArg(1, d_dprev_p);
            kernelDPow.setArg(2, order);
            kernelDPow.setArg(3, (int)(prev_rows * prev_cols));
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelDPow, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_rows * prev_cols), local_1d));

            // Calculate d(prev_act)
            cl::Buffer d_dprev_act(clContext, CL_MEM_READ_WRITE, prev_rows * prev_cols * sizeof(float));
            size_t prev_dot_size = dotProds[layer-1].size() * dotProds[layer-1][0].size();

            // Use single-work-group kernel for small sizes
            kernelReluDer.setArg(0, d_dotProds[layer-1]);
            kernelReluDer.setArg(1, d_dprev_act);
            kernelReluDer.setArg(2, (int)prev_dot_size);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelReluDer, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_dot_size), local_1d));

            // outgoing = (dL/dz_l * C^T) . d(prev_p) . d(prev_act)
            kernelHadamard2.setArg(0, d_grad_x_CT);
            kernelHadamard2.setArg(1, d_dprev_p);
            kernelHadamard2.setArg(2, d_dprev_act);
            kernelHadamard2.setArg(3, d_outgoing[layer-1]);
            kernelHadamard2.setArg(4, prev_rows);
            kernelHadamard2.setArg(5, prev_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelHadamard2, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_rows * prev_cols), local_1d));

            // --- Calculate Weight Gradients ---
            // gradc = ALPHA * prev_p^T * incoming
            // power prev_p
            cl::Buffer d_prev_p(clContext, CL_MEM_READ_WRITE, prev_rows * prev_cols * sizeof(float));
            kernelPower.setArg(0, d_activate[layer-1]);
            kernelPower.setArg(1, d_prev_p);
            kernelPower.setArg(2, order);
            kernelPower.setArg(3, (int)(prev_rows * prev_cols));
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelPower, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_rows * prev_cols), local_1d));
            // transpose prev_p
            cl::Buffer d_prev_p_T(clContext, CL_MEM_READ_WRITE, prev_cols * prev_rows * sizeof(float));
            kernelTranspose.setArg(0, d_prev_p);
            kernelTranspose.setArg(1, d_prev_p_T);
            kernelTranspose.setArg(2, prev_rows);
            kernelTranspose.setArg(3, prev_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, calculate_global_2d(size2d, prev_rows, prev_cols), local_2d));
            // dL/dC_layer
            kernelMatMul.setArg(0, d_prev_p_T);
            kernelMatMul.setArg(1, d_incoming[layer]);
            kernelMatMul.setArg(2, d_gradC[layer]);
            kernelMatMul.setArg(3, prev_cols);
            kernelMatMul.setArg(4, prev_rows);
            kernelMatMul.setArg(5, curr_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, calculate_global_2d(size2d, prev_cols, curr_cols), local_2d));
            // scale dL/dC_layer by ALPHA
            kernelScale.setArg(0, d_gradC[layer]);
            kernelScale.setArg(1, d_gradC[layer]);
            kernelScale.setArg(2, ALPHA);
            kernelScale.setArg(3, (int)(prev_cols * curr_cols));
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_cols * curr_cols), local_1d));

            // gradB = dL/dz_l x V1^T
            // transpose ones = onesT
            std::vector<float> ones(prev_cols * prev_rows, 1.0f);
            cl::Buffer d_onesT(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ones.size(), ones.data(), &err); CL_CHECK(err);
            // dL/dB_layer
            kernelMatMul.setArg(0, d_onesT);
            kernelMatMul.setArg(1, d_incoming[layer]);
            kernelMatMul.setArg(2, d_gradB[layer]);
            kernelMatMul.setArg(3, prev_cols);
            kernelMatMul.setArg(4, prev_rows);
            kernelMatMul.setArg(5, curr_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, calculate_global_2d(size2d, prev_cols, curr_cols), local_2d));
            // scale dL/dB_layer by 1- ALPHA
            kernelScale.setArg(0, d_gradB[layer]);
            kernelScale.setArg(1, d_gradB[layer]);
            kernelScale.setArg(2, 1.0f - ALPHA);
            kernelScale.setArg(3, (int)(prev_cols * curr_cols));
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_cols * curr_cols), local_1d));
        }

        // --- Backpropagation for the first layer ---
        int inHeight = input.size();
        int inWidth = input[0].size();
        int firstLayerRows = activate[0].size();
        int firstLayerCols = activate[0][0].size();

        cl::Buffer d_input(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inHeight * inWidth, (void*)flatten(input).data(), &err); CL_CHECK(err);
        cl::Buffer d_input_p(clContext, CL_MEM_READ_WRITE, inHeight * inWidth * sizeof(float));
        kernelPower.setArg(0, d_input);
        kernelPower.setArg(1, d_input_p);
        kernelPower.setArg(2, order);
        kernelPower.setArg(3, (int)(inHeight * inWidth));
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelPower, cl::NullRange, calculate_global_1d(WORKSIZE_1D, inHeight * inWidth), local_1d));

        cl::Buffer d_input_p_T(clContext, CL_MEM_READ_WRITE, inWidth * inHeight * sizeof(float));
        kernelTranspose.setArg(0, d_input_p);
        kernelTranspose.setArg(1, d_input_p_T);
        kernelTranspose.setArg(2, inHeight);
        kernelTranspose.setArg(3, inWidth);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, calculate_global_2d(size2d, inHeight, inWidth), local_2d));

        // dL/dC_1
        kernelMatMul.setArg(0, d_input_p_T);
        kernelMatMul.setArg(1, d_incoming[0]);
        kernelMatMul.setArg(2, d_gradC[0]);
        kernelMatMul.setArg(3, inWidth);
        kernelMatMul.setArg(4, inHeight);
        kernelMatMul.setArg(5, firstLayerCols);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, calculate_global_2d(size2d, inWidth, firstLayerCols), local_2d));
        // scale by ALPHA
        kernelScale.setArg(0, d_gradC[0]);
        kernelScale.setArg(1, d_gradC[0]);
        kernelScale.setArg(2, ALPHA);
        kernelScale.setArg(3, (int)(inWidth * firstLayerCols));
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, calculate_global_1d(WORKSIZE_1D, inWidth * firstLayerCols), local_1d));

        // dL/dB_1
        std::vector<float> ones(inWidth * inHeight, 1.0f);
        cl::Buffer d_ones_T(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ones.size(), ones.data(), &err); CL_CHECK(err);
        kernelMatMul.setArg(0, d_ones_T);
        kernelMatMul.setArg(1, d_incoming[0]);
        kernelMatMul.setArg(2, d_gradB[0]);
        kernelMatMul.setArg(3, inWidth);
        kernelMatMul.setArg(4, inHeight);
        kernelMatMul.setArg(5, firstLayerCols);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, calculate_global_2d(size2d, inWidth, firstLayerCols), local_2d));
        // scale by 1
        kernelScale.setArg(0, d_gradB[0]);
        kernelScale.setArg(1, d_gradB[0]);
        kernelScale.setArg(2, 1.0f - ALPHA);
        kernelScale.setArg(3, (int)(inWidth * firstLayerCols));
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, calculate_global_1d(WORKSIZE_1D, inWidth * firstLayerCols), local_1d));

        // Read the updated weights back to host
        for (int i = 0; i < this->layers; ++i) {
            size_t cweight_size = cweights[i].size() * cweights[i][0].size();
            size_t bweight_size = bweights[i].size() * bweights[i][0].size();
            std::vector<float> takeIn(cweight_size, 0.0f);
            std::vector<float> takeIn2(bweight_size, 0.0f);
            std::vector<float> takeIn3(cweight_size, 0.0f);
            std::vector<float> takeIn4(bweight_size, 0.0f);
            cl::NDRange globalCWeightGrad = calculate_global_1d(WORKSIZE_1D, cweight_size);
            cl::NDRange globalBWeightGrad = calculate_global_1d(WORKSIZE_1D, bweight_size);

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
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdateWeights, cl::NullRange, globalCWeightGrad, local_1d));
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
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdateWeights, cl::NullRange, globalBWeightGrad, local_1d));

            // copy and reshape
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_cweights[i], CL_TRUE, 0, sizeof(float) * cweight_size, (void*)takeIn.data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_bweights[i], CL_TRUE, 0, sizeof(float) * bweight_size, (void*)takeIn2.data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_gradC[i], CL_TRUE, 0, sizeof(float) * cweight_size, (void*)takeIn3.data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_gradB[i], CL_TRUE, 0, sizeof(float) * bweight_size, (void*)takeIn4.data()));
            cweights[i] = reshape(takeIn, cweights[i].size(), cweights[i][0].size());
            bweights[i] = reshape(takeIn2, bweights[i].size(), bweights[i][0].size());
            cgradients[i] = reshape(takeIn3, cweights[i].size(), cgradients[i][0].size());
            bgradients[i] = reshape(takeIn4, bweights[i].size(), bgradients[i][0].size());
            std::vector<float> acti(activate[i].size() * activate[i][0].size());
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[i], CL_TRUE, 0, sizeof(float) * acti.size(), (void*)acti.data()));
            activate[i] = reshape(acti, activate[i].size(), activate[i][0].size());
        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn2d::clBackprop: ") + e.what());
    }
}

#endif