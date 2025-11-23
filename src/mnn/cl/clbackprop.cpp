#ifdef USE_CL
#include "mnn.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <algorithm> // For std::copy
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
        std::vector<cl::Buffer> d_dpow(this->layers-1);
        std::vector<cl::Buffer> d_dact(this->layers-1);
        std::vector<cl::Buffer> d_preoutgoing(this->layers-1);
        std::vector<cl::Buffer> d_outgoing(this->layers-1);

        auto kernelSigmoidDer = kernels.at("sigmoidDer");
        auto kernelSub = kernels.at("subtract");
        auto kernelScale = kernels.at("scaleByValue");
        auto kernelDPow = kernels.at("dPower");
        auto kernelvxv2m = kernels.at("vecxvec2mat");
        auto kernelvxv2v = kernels.at("vecxvec2vec");
        auto kernelTranspose = kernels.at("transpose");
        auto kernelHadamard = kernels.at("hadamard2");
        auto kernelUpdateWeights = kernels.at("kernelUpdateWeightsElasticNet");

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
            d_gradC[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * cweight_size); CL_CHECK(err);
            d_gradB[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * bweight_size); CL_CHECK(err);
            // fill d_act and d_dotProds
            d_activate[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * activate[i].size(), (void*)activate[i].data(), &err); CL_CHECK(err);
            d_dotProds[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * dotProds[i].size(), (void*)dotProds[i].data(), &err); CL_CHECK(err);
            // allot space to gradients
            d_gradC[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * cweights[i].size() * cweights[i][0].size()); CL_CHECK(err);
            d_gradB[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * bweights[i].size() * bweights[i][0].size()); CL_CHECK(err);
            // for incoming gradients
            d_incoming[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * activate[i].size()); CL_CHECK(err);
        }
        // for outgoing gradients: from 0 to layers - 2
        for(int i = 0; i < this->layers-1; i++) {
            d_preoutgoing[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * activate[i].size()); CL_CHECK(err);
            d_outgoing[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * activate[i].size()); CL_CHECK(err);
            d_dpow[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * activate[i].size()); CL_CHECK(err);
            d_dact[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * activate[i].size()); CL_CHECK(err);
        }
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
            size_t cweight_size = cweights[layer].size() * cweights[layer][0].size();
            size_t bweight_size = bweights[layer].size() * bweights[layer][0].size();

            cl::NDRange globalWeightGrad = calculate_global_1d(WORKSIZE_1D, cweights[layer].size() * cweights[layer][0].size());
            cl::NDRange globalOutGrad = calculate_global_1d(WORKSIZE_1D, activate[layer - 1].size());

            // dL/dC_l using kernelvxv2m
            kernelvxv2m.setArg(0, d_incoming[layer]);
            kernelvxv2m.setArg(1, d_activate[layer - 1]);
            kernelvxv2m.setArg(2, d_gradC[layer]);
            kernelvxv2m.setArg(3, (int)activate[layer - 1].size());
            kernelvxv2m.setArg(4, (int)cweights[layer].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2m, cl::NullRange, globalWeightGrad, local_1d));
            // scale gradc by alpha
            kernelScale.setArg(0, d_gradC[layer]);
            kernelScale.setArg(1, d_gradC[layer]);
            kernelScale.setArg(2, alpha);
            kernelScale.setArg(3, (int)cweight_size);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGrad, local_1d));

            // dL/dB_l using kernelvxv2m
            cl::Buffer ones;
            std::vector<float> v1(activate[layer - 1].size(), 1.0f);
            ones = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * v1.size(), v1.data(), &err); CL_CHECK(err);
            kernelvxv2m.setArg(0, d_incoming[layer]);
            kernelvxv2m.setArg(1, ones);
            kernelvxv2m.setArg(2, d_gradB[layer]);
            kernelvxv2m.setArg(3, (int)bweights[layer].size());
            kernelvxv2m.setArg(4, (int)bweights[layer][0].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2m, cl::NullRange, globalWeightGrad, local_1d));
            // scale gradb by 1-alpha
            kernelScale.setArg(0, d_gradB[layer]);
            kernelScale.setArg(1, d_gradB[layer]);
            kernelScale.setArg(2, 1.0f - alpha);
            kernelScale.setArg(3, (int)bweight_size);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGrad, local_1d));

            // transpose
            cl::Buffer d_C_T(clContext, CL_MEM_READ_WRITE, cweights[layer].size() * cweights[layer][0].size() * sizeof(float));
            kernelTranspose.setArg(0, d_cweights[layer]);
            kernelTranspose.setArg(1, d_C_T);
            kernelTranspose.setArg(2, (int)cweights[layer].size());
            kernelTranspose.setArg(3, (int)cweights[layer][0].size());
            cl::NDRange globalTranspose = calculate_global_2d(size2d, (int)cweights[layer].size(), (int)cweights[layer][0].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, globalTranspose, local_2d));
            // incoming gradient x C^T
            kernelvxv2v.setArg(0, d_incoming[layer]);
            kernelvxv2v.setArg(1, d_cweights[layer]);
            kernelvxv2v.setArg(2, d_preoutgoing[layer - 1]);
            kernelvxv2v.setArg(3, (int)activate[layer - 1].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2v, cl::NullRange, globalOutGrad, local_1d));
            // derivative of power
            kernelDPow.setArg(0, d_activate[layer - 1]);
            kernelDPow.setArg(1, d_dpow[layer - 1]);
            kernelDPow.setArg(2, order);
            kernelDPow.setArg(3, (int)activate[layer - 1].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2v, cl::NullRange, globalOutGrad, local_1d));
            // derivative of activation
            kernelSigmoidDer.setArg(0, d_dotProds[layer - 1]);
            kernelSigmoidDer.setArg(1, d_dact[layer - 1]);
            kernelSigmoidDer.setArg(2, (int)dotProds[layer - 1].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSigmoidDer, cl::NullRange, globalOutGrad, local_1d));
            // outgoing gradient = (dl/dz_l x C^T) . dprev_p . dprevAct
            // kernel hadamard to perform element-wise multiplication of each component
            kernelHadamard.setArg(0, d_preoutgoing[layer - 1]);
            kernelHadamard.setArg(1, d_dpow[layer - 1]);
            kernelHadamard.setArg(2, d_dact[layer - 1]);
            kernelHadamard.setArg(3, d_outgoing[layer - 1]);
            kernelHadamard.setArg(4, (int)activate[layer - 1].size());
            kernelHadamard.setArg(5, (int)1);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelHadamard, cl::NullRange, globalOutGrad, local_1d));
            CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_outgoing[layer - 1], d_incoming[layer - 1], 0, 0, sizeof(float) * activate[layer - 1].size()));
        }

        // Backpropagation for the first layer (input layer)
        cl::NDRange globalWeightGradFirst = calculate_global_1d(WORKSIZE_1D, cweights[0].size() * cweights[0][0].size());
        size_t cweight_size = cweights[0].size() * cweights[0][0].size();
        size_t bweight_size = bweights[0].size() * bweights[0][0].size();
        cl::Buffer d_ones(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), std::vector<float>(input.size(), 1.0f).data(), &err); CL_CHECK(err);
        // dL/dC_1
        kernelvxv2m.setArg(0, d_incoming[0]);
        kernelvxv2m.setArg(1, d_in);
        kernelvxv2m.setArg(2, d_gradC[0]);
        kernelvxv2m.setArg(3, (int)bweights[0].size());
        kernelvxv2m.setArg(4, (int)bweights[0][0].size());
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2m, cl::NullRange, globalWeightGradFirst, local_1d));
        // scale gradc by alpha
        kernelScale.setArg(0, d_gradC[0]);
        kernelScale.setArg(1, d_gradC[0]);
        kernelScale.setArg(2, alpha);
        kernelScale.setArg(3, (int)cweight_size);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGradFirst, local_1d));

        // dL/dB_1
        kernelvxv2m.setArg(0, d_incoming[0]);
        kernelvxv2m.setArg(1, d_ones);
        kernelvxv2m.setArg(2, d_gradB[0]);
        kernelvxv2m.setArg(3, (int)input.size());
        kernelvxv2m.setArg(4, (int)bweights[0][0].size());
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2m, cl::NullRange, globalWeightGradFirst, local_1d));
        // scale gradb by 1-alpha
        kernelScale.setArg(0, d_gradB[0]);
        kernelScale.setArg(1, d_gradB[0]);
        kernelScale.setArg(2, 1.0f);
        kernelScale.setArg(3, (int)bweight_size);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGradFirst, local_1d));

        // update weights and copy to host vectors
        for (int i = 0; i < this->layers; ++i) {
            size_t cweight_size = cweights[i].size() * cweights[i][0].size();
            size_t bweight_size = bweights[i].size() * bweights[i][0].size();
            // flat vectors for C, B weights and gradients
            std::vector<float> takeIn(cweight_size, 0.0f);
            std::vector<float> takeIn2(bweight_size, 0.0f);
            std::vector<float> takeIn3(cweight_size, 0.0f);
            std::vector<float> takeIn4(bweight_size, 0.0f);
            cl::NDRange globalWeightGrad = calculate_global_1d(WORKSIZE_1D, cweights[i].size() * cweights[i][0].size());

            // Update C weights using kernelUpdateWeights
            kernelUpdateWeights.setArg(0, d_cweights[i]);
            kernelUpdateWeights.setArg(1, d_gradC[i]);
            kernelUpdateWeights.setArg(2, (int)cweight_size);
            kernelUpdateWeights.setArg(3, learningRate);
            kernelUpdateWeights.setArg(4, LAMBDA_L1);
            kernelUpdateWeights.setArg(5, LAMBDA_L2);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdateWeights, cl::NullRange, globalWeightGrad, local_1d));
            // Update B weights using kernelUpdateWeights
            kernelUpdateWeights.setArg(0, d_bweights[i]);
            kernelUpdateWeights.setArg(1, d_gradB[i]);
            kernelUpdateWeights.setArg(2, (int)bweight_size);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdateWeights, cl::NullRange, globalWeightGrad, local_1d));

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
        auto kernelSoftmaxDer = kernels.at("softmaxDer");
        auto kernelHadamard2 = kernels.at("hadamard2");
        auto kernelMatMul = kernels.at("matxmat2mat");
        auto kernelScale = kernels.at("scaleByValue");
        auto kernelUpdateWeights = kernels.at("kernelUpdateWeightsElasticNet");
        auto kernelTranspose = kernels.at("transpose");
        auto kernelDPow = kernels.at("dPower");
        auto kernelPower = kernels.at("power");

        for (int i = 0; i < this->layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            size_t dot_size = dotProds[i].size() * dotProds[i][0].size();
            size_t act_size = activate[i].size() * activate[i][0].size();

            d_cweights[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, c_size * sizeof(float), (void*)flatten(cweights[i]).data(), &err); CL_CHECK(err);
            d_bweights[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, b_size * sizeof(float), (void*)flatten(bweights[i]).data(), &err); CL_CHECK(err);
            d_dotProds[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dot_size * sizeof(float), (void*)flatten(dotProds[i]).data(), &err); CL_CHECK(err);
            d_activate[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, act_size * sizeof(float), (void*)flatten(activate[i]).data(), &err); CL_CHECK(err);
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

        // --- Backpropagation ---
        // Initial error (output - expected)
        kernelSub.setArg(0, d_out);
        kernelSub.setArg(1, d_exp);
        kernelSub.setArg(2, d_err);
        kernelSub.setArg(3, (int)output.size());
        cl::NDRange globalSub = calculate_global_1d(WORKSIZE_1D, output.size());
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSub, cl::NullRange, globalSub, local_1d));
        CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_err, d_incoming[layers - 1], 0, 0, sizeof(float) * output.size()));

        // error back through mean pooling layer
        std::vector<std::vector<float>> equalGrads(activate[layers-1].size(), std::vector<float>(activate[layers-1][0].size()));
        std::vector<float> out_err_host(output.size());
        CL_CHECK(clCommandQueue.enqueueReadBuffer(d_err, CL_TRUE, 0, sizeof(float) * output.size(), out_err_host.data()));
        for(size_t i = 0; i < activate[layers-1].size(); ++i) {
            for(size_t j = 0; j < activate[layers-1][0].size(); ++j) {
                equalGrads[i][j] = out_err_host[j];
            }
        }
        std::vector<float> last_layer_err = flatten(equalGrads);
        CL_CHECK(clCommandQueue.enqueueWriteBuffer(d_incoming[layers-1], CL_TRUE, 0, last_layer_err.size() * sizeof(float), last_layer_err.data()));

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

            if (prev_dot_size > WORKSIZE_1D) {
                // Use multi-work-group approach for large sizes
                cl::Kernel kernelSoftMaxReduce = kernels.at("softmax_reduce");
                cl::Kernel kernelSoftMaxDerNormalize = kernels.at("softmaxDer_normalize");

                size_t num_work_groups = (prev_dot_size + WORKSIZE_1D - 1) / WORKSIZE_1D;
                size_t partial_results_buffer_size = num_work_groups * 2;

                cl::Buffer d_partial_results(clContext, CL_MEM_READ_WRITE, sizeof(float) * partial_results_buffer_size, nullptr, &err); CL_CHECK(err);

                // Launch softmax_reduce kernel (same as in forprop)
                kernelSoftMaxReduce.setArg(0, d_dotProds[layer-1]);
                kernelSoftMaxReduce.setArg(1, d_partial_results);
                kernelSoftMaxReduce.setArg(2, cl::Local(sizeof(float) * WORKSIZE_1D));
                kernelSoftMaxReduce.setArg(3, cl::Local(sizeof(float) * WORKSIZE_1D));
                kernelSoftMaxReduce.setArg(4, (int)prev_dot_size);
                kernelSoftMaxReduce.setArg(5, SOFTMAX_TEMP);
                cl::NDRange globalReduce = calculate_global_1d(WORKSIZE_1D, prev_dot_size);
                cl::NDRange localReduce(WORKSIZE_1D);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSoftMaxReduce, cl::NullRange, globalReduce, localReduce));

                std::vector<float> h_partial_results(partial_results_buffer_size);
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_partial_results, CL_TRUE, 0, sizeof(float) * partial_results_buffer_size, h_partial_results.data()));

                float global_max = -(std::numeric_limits<float>::max)();
                float global_sum = 0.0f;
                for (size_t k = 0; k < num_work_groups; ++k) {
                    global_sum += h_partial_results[2 * k];
                    global_max = (std::max)(global_max, h_partial_results[2 * k + 1]);
                }

                // Launch the new softmaxDer_normalize kernel
                kernelSoftMaxDerNormalize.setArg(0, d_dotProds[layer-1]);
                kernelSoftMaxDerNormalize.setArg(1, d_dprev_act);
                kernelSoftMaxDerNormalize.setArg(2, (int)prev_dot_size);
                kernelSoftMaxDerNormalize.setArg(3, SOFTMAX_TEMP);
                kernelSoftMaxDerNormalize.setArg(4, global_max);
                kernelSoftMaxDerNormalize.setArg(5, global_sum);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSoftMaxDerNormalize, cl::NullRange, globalReduce, localReduce));
            }
            else {
                // Use single-work-group kernel for small sizes
                kernelSoftmaxDer.setArg(0, d_dotProds[layer-1]);
                kernelSoftmaxDer.setArg(1, d_dprev_act);
                kernelSoftmaxDer.setArg(2, SOFTMAX_TEMP);
                kernelSoftmaxDer.setArg(3, (int)prev_dot_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSoftmaxDer, cl::NullRange, cl::NDRange(prev_dot_size), cl::NDRange(prev_dot_size)));
            }

            // outgoing = (dL/dz_l * C^T) . d(prev_p) . d(prev_act)
            kernelHadamard2.setArg(0, d_grad_x_CT);
            kernelHadamard2.setArg(1, d_dprev_p);
            kernelHadamard2.setArg(2, d_dprev_act);
            kernelHadamard2.setArg(3, d_outgoing[layer-1]);
            kernelHadamard2.setArg(4, prev_rows);
            kernelHadamard2.setArg(5, prev_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelHadamard2, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_rows * prev_cols), local_1d));

            // --- Calculate Weight Gradients ---
            // gradc = alpha * prev_p^T * incoming
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
            // scale dL/dC_layer by alpha
            kernelScale.setArg(0, d_gradC[layer]);
            kernelScale.setArg(1, d_gradC[layer]);
            kernelScale.setArg(2, alpha);
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
            // scale dL/dB_layer by 1- alpha
            kernelScale.setArg(0, d_gradB[layer]);
            kernelScale.setArg(1, d_gradB[layer]);
            kernelScale.setArg(2, 1.0f - alpha);
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
        // scale by alpha
        kernelScale.setArg(0, d_gradC[0]);
        kernelScale.setArg(1, d_gradC[0]);
        kernelScale.setArg(2, alpha);
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
        kernelScale.setArg(2, 1.0f - alpha);
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
            cl::NDRange globalWeightGrad = calculate_global_1d(WORKSIZE_1D, cweights[i].size() * cweights[i][0].size());
            size_t prev_layer_size = (i == 0) ? input.size() : activate[i - 1].size();

            // Update C weights using kernelUpdateWeights
            kernelUpdateWeights.setArg(0, d_cweights[i]);
            kernelUpdateWeights.setArg(1, d_gradC[i]);
            kernelUpdateWeights.setArg(2, (int)cweights[i].size() * (int)prev_layer_size);
            kernelUpdateWeights.setArg(3, learningRate);
            kernelUpdateWeights.setArg(4, LAMBDA_L1);
            kernelUpdateWeights.setArg(5, LAMBDA_L2);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdateWeights, cl::NullRange, globalWeightGrad, local_1d));
            // Update B weights using kernelUpdateWeights
            kernelUpdateWeights.setArg(0, d_bweights[i]);
            kernelUpdateWeights.setArg(1, d_gradB[i]);
            kernelUpdateWeights.setArg(2, (int)bweights[i].size() * (int)prev_layer_size);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdateWeights, cl::NullRange, globalWeightGrad, local_1d));

            // copy and reshape
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_cweights[i], CL_TRUE, 0, sizeof(float) * cweight_size, (void*)takeIn.data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_bweights[i], CL_TRUE, 0, sizeof(float) * bweight_size, (void*)takeIn2.data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_gradC[i], CL_TRUE, 0, sizeof(float) * cweight_size, (void*)takeIn3.data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_gradB[i], CL_TRUE, 0, sizeof(float) * bweight_size, (void*)takeIn4.data()));
            cweights[i] = reshape(takeIn, cweights[i].size(), cweights[i][0].size());
            bweights[i] = reshape(takeIn2, bweights[i].size(), bweights[i][0].size());
            cgradients[i] = reshape(takeIn3, cweights[i].size(), cgradients[i][0].size());
            bgradients[i] = reshape(takeIn4, bweights[i].size(), bgradients[i][0].size());
        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn2d::clBackprop: ") + e.what());
    }
}

#endif