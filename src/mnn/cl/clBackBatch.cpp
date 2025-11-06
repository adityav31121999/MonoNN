#ifdef USE_OPENCL
#include "mnn.hpp"
#include <vector>
#include <iostream>
#include <stdexcept>

/**
 * @brief batch backpropgation using opencl for mnn
 * @param expected expected result from forprop
 */
void mnn::clBackprop(const std::vector<std::vector<float>>& expected) {
    try {
        cl_int err;
        int batchSize = expected.size();
        if (batchSize == 0) {
            throw std::runtime_error("Batch size should be greater than 0.");
        }
        cl::NDRange local_1d(WORKSIZE_1D);
        cl::NDRange local_2d(WORKSIZE_2DX, WORKSIZE_2DY);
        size_t size2d[2] = {WORKSIZE_2DX, WORKSIZE_2DY};

        size_t inputSize = input.size();
        size_t outputSize = output.size();

        std::vector<cl::Buffer> d_in, d_exp, d_out, d_err;
        std::vector<std::vector<cl::Buffer>> d_incoming(this->layers);
        std::vector<cl::Buffer> d_cweights(this->layers);
        std::vector<cl::Buffer> d_bweights(this->layers);
        std::vector<cl::Buffer> d_gradC(this->layers);
        std::vector<cl::Buffer> d_gradB(this->layers);
        std::vector<std::vector<cl::Buffer>> d_dotProds(this->layers);
        std::vector<std::vector<cl::Buffer>> d_activate(this->layers);
        std::vector<std::vector<cl::Buffer>> d_dpow(this->layers-1);
        std::vector<std::vector<cl::Buffer>> d_dact(this->layers-1);
        std::vector<std::vector<cl::Buffer>> d_preoutgoing(this->layers-1);
        std::vector<std::vector<cl::Buffer>> d_outgoing(this->layers-1);

        cl::Kernel kernelSigmoidDer, kernelSub, kernelDPow, kernelvxv2m, kernelvxv2v, kernelUpdateWeights, kernelHadamard, kernelTranspose, kernelAvg;
        kernelSigmoidDer = kernels.at("sigmoidDer");
        kernelSub = kernels.at("subtract");
        kernelDPow = kernels.at("dPower");
        kernelvxv2m = kernels.at("vecxvec2mat");
        kernelvxv2v = kernels.at("vecxvec2vec");
        kernelTranspose = kernels.at("transpose");
        kernelHadamard = kernels.at("hadamard2");
        kernelDPow = kernels.at("dPower");
        kernelAvg = kernels.at("matrix_vector_average");
        kernelUpdateWeights = kernels.at("kernelUpdateWeightsElasticNet");

        // buffer data and space allotment
        for (int i = 0; i < batchSize; i++) {
            d_in[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputSize, inputBatch[i].data(), &err); CL_CHECK(err);
            d_exp[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * expected[i].size(), (void*)expected.data(), &err); CL_CHECK(err);
            d_out[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * outputBatch[i].size(), (void*)outputBatch[i].data(), &err); CL_CHECK(err);
            d_err[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * outputBatch[i].size()); CL_CHECK(err);
        }
        for(int i = 0; i < this->layers; i++) {
            // allot weights to c and b
            size_t cweight_size = cweights[i].size() * cweights[i][0].size();
            size_t bweight_size = bweights[i].size() * bweights[i][0].size();
            d_cweights[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * cweight_size, (void*)flatten(cweights[i]).data(), &err); CL_CHECK(err);
            d_bweights[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * bweight_size, (void*)flatten(bweights[i]).data(), &err); CL_CHECK(err);
            d_gradC[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * cweight_size); CL_CHECK(err);
            d_gradB[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * bweight_size); CL_CHECK(err);
            // fill d_act and d_dotProds
            for(int j = 0; j < batchSize; j++) {
                d_activate[i][j] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * actBatch[i][j].size(), (void*)actBatch[i][j].data(), &err); CL_CHECK(err);
                d_dotProds[i][j] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * dotBatch[i][j].size(), (void*)dotBatch[i][j].data(), &err); CL_CHECK(err);
                d_incoming[i][j] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * actBatch[i][j].size()); CL_CHECK(err);
            }
        }
        for(int i = 0; i < this->layers-1; i++) {
            size_t actSize = actBatch[i].size() * actBatch[i][0].size();
            d_preoutgoing[i] = std::vector<cl::Buffer>(batchSize, cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * actSize)); CL_CHECK(err);
            d_outgoing[i] = std::vector<cl::Buffer>(batchSize, cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * actSize)); CL_CHECK(err);
            d_dpow[i] = std::vector<cl::Buffer>(batchSize, cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * actSize)); CL_CHECK(err);
            d_dact[i] = std::vector<cl::Buffer>(batchSize, cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * actSize)); CL_CHECK(err);
        }

        for(int i = 0; i < batchSize; i++) {
            // Calculate initial error (output - expected)
            kernelSub.setArg(0, d_out[i]);
            kernelSub.setArg(1, d_exp[i]);
            kernelSub.setArg(2, d_err[i]);
            kernelSub.setArg(3, (int)outputSize);
            cl::NDRange globalSub = calculate_global_1d(WORKSIZE_1D, outputSize);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSub, cl::NullRange, globalSub, local_1d));
            CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_err[i], d_incoming[layers - 1][i], 0, 0, sizeof(float) * outputSize));
        }

        for(int layer = layers - 1; layers <= 1; --layer) {
            cl::NDRange globalWeightGrad = calculate_global_1d(WORKSIZE_1D, cweights[layer].size() * cweights[layer][0].size());
            cl::NDRange globalOutGrad = calculate_global_1d(WORKSIZE_1D, actBatch[layer - 1][0].size());
            int row = cgradients[layer].size();
            int col = cgradients[layer][0].size();
            int totalElements = row * col;
            cl::Buffer d_totalCgrad, d_totalBgrad;
            d_totalCgrad = cl::Buffer(clContext, CL_MEM_READ_WRITE, totalElements * batchSize * sizeof(float));
            d_totalBgrad = cl::Buffer(clContext, CL_MEM_READ_WRITE, totalElements * batchSize * sizeof(float));

            for(int i = 0; i < batchSize; ++i) {
                // step wise backward propgation
                // dL/dC_l using kernelvxv2m
                kernelvxv2m.setArg(0, d_incoming[layer][i]);
                kernelvxv2m.setArg(1, d_activate[layer - 1][i]);
                kernelvxv2m.setArg(2, d_gradC[layer]);
                kernelvxv2m.setArg(3, (int)cweights[layer].size());
                kernelvxv2m.setArg(4, (int)cweights[layer][0].size());
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2m, cl::NullRange, globalWeightGrad, local_1d));
                CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_gradC[i], d_totalCgrad, 0, i * totalElements * sizeof(float), sizeof(float) * totalElements));

                // dL/dB_l using kernelvxv2m
                cl::Buffer ones;
                std::vector<float> v1(actBatch[layer - 1][i].size(), 1.0f);
                ones = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * v1.size(), v1.data(), &err); CL_CHECK(err);
                kernelvxv2m.setArg(0, d_incoming[layer][i]);
                kernelvxv2m.setArg(1, ones);
                kernelvxv2m.setArg(2, d_gradB[layer]);
                kernelvxv2m.setArg(3, (int)bweights[layer].size());
                kernelvxv2m.setArg(4, (int)bweights[layer][0].size());
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2m, cl::NullRange, globalWeightGrad, local_1d));
                CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_gradB[i], d_totalBgrad, 0, i * totalElements * sizeof(float), sizeof(float) * totalElements));

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
                kernelvxv2v.setArg(2, d_preoutgoing[layer - 1][i]);
                kernelvxv2v.setArg(3, (int)actBatch[layer - 1][i].size());
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2v, cl::NullRange, globalOutGrad, local_1d));
                // derivative of power
                kernelDPow.setArg(0, d_activate[layer - 1]);
                kernelDPow.setArg(1, d_dpow[layer - 1]);
                kernelDPow.setArg(2, order);
                kernelDPow.setArg(3, (int)actBatch[layer - 1][i].size());
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2v, cl::NullRange, globalOutGrad, local_1d));
                // derivative of activation
                kernelSigmoidDer.setArg(0, d_dotProds[layer - 1]);
                kernelSigmoidDer.setArg(1, d_dact[layer - 1]);
                kernelSigmoidDer.setArg(2, (int)dotBatch[layer - 1][i].size());
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSigmoidDer, cl::NullRange, globalOutGrad, local_1d));
                // outgoing gradient = (dl/dz_l x C^T) . dprev_p . dprevAct
                kernelHadamard.setArg(0, d_preoutgoing[layer - 1][i]);
                kernelHadamard.setArg(1, d_dpow[layer - 1][i]);
                kernelHadamard.setArg(2, d_dact[layer - 1][i]);
                kernelHadamard.setArg(3, d_outgoing[layer - 1][i]);
                kernelHadamard.setArg(4, (int)actBatch[layer - 1][i].size());
                kernelHadamard.setArg(5, (int)1);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelHadamard, cl::NullRange, globalOutGrad, local_1d));
            }

            // average the gradients
            kernelAvg.setArg(0, d_totalCgrad);
            kernelAvg.setArg(1, d_gradC[layer]);
            kernelAvg.setArg(2, batchSize);
            kernelAvg.setArg(3, cgradients[layer].size());
            kernelAvg.setArg(4, cgradients[layer][0].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, globalWeightGrad, local_1d));
            kernelAvg.setArg(0, d_totalBgrad);
            kernelAvg.setArg(1, d_gradB[layer]);
            kernelAvg.setArg(2, batchSize);
            kernelAvg.setArg(3, bgradients[layer].size());
            kernelAvg.setArg(4, bgradients[layer][0].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, globalWeightGrad, local_1d));
        }

        // for first layer
        cl::NDRange globalWeightGradFirst = calculate_global_1d(WORKSIZE_1D, cweights[0].size() * cweights[0][0].size());
        cl::NDRange globalOutGrad = calculate_global_1d(WORKSIZE_1D, actBatch[0][0].size());
        int row = cgradients[0].size();
        int col = cgradients[0][0].size();
        int totalElements = row * col;
        cl::Buffer d_totalCgrad, d_totalBgrad;
        d_totalCgrad = cl::Buffer(clContext, CL_MEM_READ_WRITE, totalElements * batchSize * sizeof(float));
        d_totalBgrad = cl::Buffer(clContext, CL_MEM_READ_WRITE, totalElements * batchSize * sizeof(float));

        for(int i = 0; i < batchSize; i++) {
            // Backpropagation for the first layer (input layer)
            // dL/dC_1
            kernelvxv2m.setArg(0, d_incoming[0][i]);
            kernelvxv2m.setArg(1, d_in[i]);
            kernelvxv2m.setArg(2, d_gradC[0]);
            kernelvxv2m.setArg(3, (int)inputBatch[i].size());
            kernelvxv2m.setArg(4, (int)cweights[0].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2m, cl::NullRange, globalWeightGradFirst, local_1d));
            CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_gradC[0], d_totalCgrad, 0, i * totalElements * sizeof(float), sizeof(float) * totalElements));

            // dL/dB_1
            kernelvxv2m.setArg(0, d_incoming[0][i]);
            kernelvxv2m.setArg(1, d_bweights[0]);
            kernelvxv2m.setArg(2, d_gradB[0]);
            kernelvxv2m.setArg(3, (int)inputBatch[i].size());
            kernelvxv2m.setArg(4, (int)bweights[0].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2m, cl::NullRange, globalWeightGradFirst, local_1d));
            CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_gradB[0], d_totalCgrad, 0, i * totalElements * sizeof(float), sizeof(float) * totalElements));
        }

        // average the gradients
        kernelAvg.setArg(0, d_totalCgrad);
        kernelAvg.setArg(1, d_gradC[0]);
        kernelAvg.setArg(2, batchSize);
        kernelAvg.setArg(3, cgradients[0].size());
        kernelAvg.setArg(4, cgradients[0][0].size());
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, globalWeightGradFirst, local_1d));
        kernelAvg.setArg(0, d_totalBgrad);
        kernelAvg.setArg(1, d_gradB[0]);
        kernelAvg.setArg(2, batchSize);
        kernelAvg.setArg(3, bgradients[0].size());
        kernelAvg.setArg(4, bgradients[0][0].size());
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, globalWeightGradFirst, local_1d));

        // Read the updated weights back to host
        for (int i = 0; i < this->layers; ++i) {
            size_t cweight_size = cweights[i].size() * cweights[i][0].size();
            size_t bweight_size = bweights[i].size() * bweights[i][0].size();
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
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_cweights[i], CL_TRUE, 0, sizeof(float) * cweight_size, (void*)takeIn.data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_bweights[i], CL_TRUE, 0, sizeof(float) * bweight_size, (void*)takeIn2.data()));
            cweights[i] = reshape(takeIn, cweights[i].size(), cweights[i][0].size());
            bweights[i] = reshape(takeIn2, bweights[i].size(), bweights[i][0].size());
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_gradC[i], CL_TRUE, 0, sizeof(float) * cweight_size, (void*)takeIn3.data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_gradB[i], CL_TRUE, 0, sizeof(float) * bweight_size, (void*)takeIn4.data()));
            cgradients[i] = reshape(takeIn3, cgradients[i].size(), cgradients[i][0].size());
            bgradients[i] = reshape(takeIn4, bweights[i].size(), bgradients[i][0].size()); 
        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn::clBackprop (batch): ") + e.what());
    }
}


/**
 * @brief batch backpropgation using opencl for mnn2d
 * @param expected expected result from forprop
 */
void mnn2d::clBackprop(const std::vector<std::vector<float>>& expected) {
    try {
        cl_int err;
        int batchSize = expected.size();
        if (batchSize == 0) {
            throw std::runtime_error("Batch size should be greater than 0.");
        }
        cl::NDRange local_1d(WORKSIZE_1D);
        cl::NDRange local_2d(WORKSIZE_2DX, WORKSIZE_2DY);
        size_t size2d[2] = {WORKSIZE_2DX, WORKSIZE_2DY};

        std::vector<cl::Buffer> d_in, d_exp, d_out, d_err;
        std::vector<std::vector<cl::Buffer>> d_incoming(this->layers);
        std::vector<cl::Buffer> d_cweights(this->layers);
        std::vector<cl::Buffer> d_bweights(this->layers);
        std::vector<cl::Buffer> d_gradC(this->layers);
        std::vector<cl::Buffer> d_gradB(this->layers);
        std::vector<std::vector<cl::Buffer>> d_dotProds(this->layers);
        std::vector<std::vector<cl::Buffer>> d_activate(this->layers);
        std::vector<std::vector<cl::Buffer>> d_dpow(this->layers-1);
        std::vector<std::vector<cl::Buffer>> d_dact(this->layers-1);
        std::vector<std::vector<cl::Buffer>> d_preoutgoing(this->layers-1);
        std::vector<std::vector<cl::Buffer>> d_outgoing(this->layers-1);

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
        auto kernelAvg = kernels.at("matrix_vector_average");

        // buffer data and space allotment
        for (int i = 0; i < batchSize; i++) {
            d_in[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inHeight * inWidth, (void*)flatten(inputBatch[i]).data(), &err); CL_CHECK(err);
            d_exp[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * expected[i].size(), (void*)expected.data(), &err); CL_CHECK(err);
            d_out[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * outputBatch[i].size(), (void*)outputBatch[i].data(), &err); CL_CHECK(err);
            d_err[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * outputBatch[i].size()); CL_CHECK(err);
        }
        for(int i = 0; i < this->layers; i++) {
            // allot weights to c and b
            size_t cweight_size = cweights[i].size() * cweights[i][0].size();
            size_t bweight_size = bweights[i].size() * bweights[i][0].size();
            d_cweights[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * cweight_size, (void*)flatten(cweights[i]).data(), &err); CL_CHECK(err);
            d_bweights[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * bweight_size, (void*)flatten(bweights[i]).data(), &err); CL_CHECK(err);
            d_gradC[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * cweight_size); CL_CHECK(err);
            d_gradB[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * bweight_size); CL_CHECK(err);
            // fill d_act and d_dotProds
            for(int j = 0; j < batchSize; j++) {
                d_activate[i][j] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * actBatch[i][j].size(), (void*)actBatch[i][j].data(), &err); CL_CHECK(err);
                d_dotProds[i][j] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * dotBatch[i][j].size(), (void*)dotBatch[i][j].data(), &err); CL_CHECK(err);
                d_incoming[i][j] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * actBatch[i][j].size()); CL_CHECK(err);
            }
        }
        for(int i = 0; i < this->layers-1; i++) {
            size_t actSize = actBatch[i][0].size() * actBatch[i][0][0].size();
            d_preoutgoing[i] = std::vector<cl::Buffer>(batchSize, cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * actSize)); CL_CHECK(err);
            d_outgoing[i] = std::vector<cl::Buffer>(batchSize, cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * actSize)); CL_CHECK(err);
            d_dpow[i] = std::vector<cl::Buffer>(batchSize, cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * actSize)); CL_CHECK(err);
            d_dact[i] = std::vector<cl::Buffer>(batchSize, cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * actSize)); CL_CHECK(err);
        }

        // error
        for (int i = 0; i < batchSize; i++) {
            kernelSub.setArg(0, d_out[i]);
            kernelSub.setArg(1, d_exp[i]);
            kernelSub.setArg(2, d_err[i]);
            kernelSub.setArg(3, (int)output.size());
            cl::NDRange globalSub = calculate_global_1d(WORKSIZE_1D, output.size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSub, cl::NullRange, globalSub, local_1d));
            CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_err[i], d_incoming[layers - 1][i], 0, 0, sizeof(float) * output.size()));

            // error back through mean pooling layer
            std::vector<std::vector<float>> equalGrads(actBatch[layers-1][i].size() * actBatch[layers-1][i][0].size());
            std::vector<float> out_err_host(output.size());
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_err[i], CL_TRUE, 0, sizeof(float) * output.size(), out_err_host.data()));
            for(size_t i = 0; i < actBatch[layers-1][i].size(); ++i) {
                for(size_t j = 0; j < actBatch[layers-1][i][0].size(); ++j) {
                    equalGrads[i][j] = out_err_host[j];
                }
            }
            std::vector<float> last_layer_err = flatten(equalGrads);
            CL_CHECK(clCommandQueue.enqueueWriteBuffer(d_incoming[layers-1][i], CL_TRUE, 0, last_layer_err.size() * sizeof(float), last_layer_err.data()));
        }

        // bacpropagation from last to second layer
        for(int layer = layers - 1; layer >= 1; layer--) {
            cl::NDRange globalWeightGrad = calculate_global_1d(WORKSIZE_1D, cweights[layer].size() * cweights[layer][0].size());
            cl::NDRange globalOutGrad = calculate_global_1d(WORKSIZE_1D, actBatch[layer - 1][0].size());
            int row = cgradients[layer].size();
            int col = cgradients[layer][0].size();
            int totalElements = row * col;
            cl::Buffer d_totalCgrad, d_totalBgrad;
            d_totalCgrad = cl::Buffer(clContext, CL_MEM_READ_WRITE, totalElements * batchSize * sizeof(float));
            d_totalBgrad = cl::Buffer(clContext, CL_MEM_READ_WRITE, totalElements * batchSize * sizeof(float));

            // step wise backpropagation
            int prev_rows = actBatch[layer-1][0].size();
            int prev_cols = actBatch[layer-1][0][0].size();
            int curr_rows = actBatch[layer][0].size();
            int curr_cols = actBatch[layer][0][0].size();

            for(int i = 0; i < batchSize; i++) {
                // transpose
                cl::Buffer d_C_T(clContext, CL_MEM_READ_WRITE, curr_cols * prev_cols * sizeof(float));
                kernelTranspose.setArg(0, d_cweights[layer]);
                kernelTranspose.setArg(1, d_C_T);
                kernelTranspose.setArg(2, prev_cols);
                kernelTranspose.setArg(3, curr_cols);
                cl::NDRange globalTranspose = calculate_global_2d(size2d, prev_cols, curr_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, globalTranspose, local_2d));

                // grad x C^T
                cl::Buffer d_grad_x_CT(clContext, CL_MEM_READ_WRITE, prev_rows * prev_cols * sizeof(float));
                kernelMatMul.setArg(0, d_incoming[layer][i]);
                kernelMatMul.setArg(1, d_C_T);
                kernelMatMul.setArg(2, d_grad_x_CT);
                kernelMatMul.setArg(3, curr_rows);
                kernelMatMul.setArg(4, curr_cols);
                kernelMatMul.setArg(5, prev_cols);
                cl::NDRange globalMatMul = calculate_global_2d(size2d, curr_rows, prev_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, globalMatMul, local_2d));

                //  d(prev_p)
                cl::Buffer d_dprev_p(clContext, CL_MEM_READ_WRITE, prev_rows * prev_cols * sizeof(float));
                kernelDPow.setArg(0, d_activate[layer-1][i]);
                kernelDPow.setArg(1, d_dprev_p);
                kernelDPow.setArg(2, order);
                kernelDPow.setArg(3, (int)(prev_rows * prev_cols));
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelDPow, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_rows * prev_cols), local_1d));

                // Calculate d(prev_act)
                cl::Buffer d_dprev_act(clContext, CL_MEM_READ_WRITE, prev_rows * prev_cols * sizeof(float));
                size_t prev_dot_size = dotBatch[layer-1][0].size() * dotBatch[layer-1][0][0].size();
                if (prev_dot_size > WORKSIZE_1D) {
                    throw std::runtime_error("Softmax derivative kernel cannot process size > WORKSIZE_1D.");
                }
                kernelSoftmaxDer.setArg(0, dotBatch[layer-1][0]);
                kernelSoftmaxDer.setArg(1, d_dprev_act);
                kernelSoftmaxDer.setArg(2, SOTMAX_TEMP);
                kernelSoftmaxDer.setArg(3, (int)prev_dot_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSoftmaxDer, cl::NullRange, cl::NDRange(prev_dot_size), cl::NDRange(prev_dot_size)));

                // outgoing = (dL/dz_l * C^T) .* d(prev_p) .* d(prev_act)
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
                // dL/dC
                kernelMatMul.setArg(0, d_prev_p_T);
                kernelMatMul.setArg(1, d_incoming[layer]);
                kernelMatMul.setArg(2, d_gradC[layer]);
                kernelMatMul.setArg(3, prev_cols);
                kernelMatMul.setArg(4, prev_rows);
                kernelMatMul.setArg(5, curr_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, calculate_global_2d(size2d, prev_cols, curr_cols), local_2d));
                // scale dL/dC by alpha
                kernelScale.setArg(0, d_gradC[layer]);
                kernelScale.setArg(1, d_gradC[layer]);
                kernelScale.setArg(2, alpha);
                kernelScale.setArg(3, (int)(prev_cols * curr_cols));
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_cols * curr_cols), local_1d));
                CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_gradC[i], d_totalCgrad, 0, i * totalElements * sizeof(float), sizeof(float) * totalElements));

                // transpose ones = onesT
                std::vector<float> ones(prev_cols * prev_rows, 1.0f);
                cl::Buffer d_ones(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ones.size(), ones.data(), &err); CL_CHECK(err);
                cl::Buffer d_onesT(clContext, CL_MEM_READ_WRITE, prev_cols * prev_rows * sizeof(float));
                kernelTranspose.setArg(0, d_ones);
                kernelTranspose.setArg(1, d_onesT);
                kernelTranspose.setArg(2, prev_rows);
                kernelTranspose.setArg(3, prev_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, calculate_global_2d(size2d, prev_rows, prev_cols), local_2d));
                // dL/dB
                kernelMatMul.setArg(0, d_onesT);
                kernelMatMul.setArg(1, d_incoming[layer]);
                kernelMatMul.setArg(2, d_gradC[layer]);
                kernelMatMul.setArg(3, prev_cols);
                kernelMatMul.setArg(4, prev_rows);
                kernelMatMul.setArg(5, curr_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, calculate_global_2d(size2d, prev_cols, curr_cols), local_2d));
                // scale dL/dB by 1.0f - alpha
                kernelScale.setArg(0, d_gradB[layer]);
                kernelScale.setArg(1, d_gradB[layer]);
                kernelScale.setArg(2, 1.0f - alpha);
                kernelScale.setArg(3, (int)(prev_cols * curr_cols));
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_cols * curr_cols), local_1d));
                CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_gradB[i], d_totalBgrad, 0, i * totalElements * sizeof(float), sizeof(float) * totalElements));

                // Copy outgoing to incoming for the next iteration
                CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_outgoing[layer-1][i], d_incoming[layer-1][i], 0, 0, sizeof(float) * prev_rows * prev_cols));
            }

            // average the gradients
            kernelAvg.setArg(0, d_totalCgrad);
            kernelAvg.setArg(1, d_gradC[layer]);
            kernelAvg.setArg(2, batchSize);
            kernelAvg.setArg(3, cgradients[layer].size());
            kernelAvg.setArg(4, cgradients[layer][0].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, globalWeightGrad, local_1d));
            kernelAvg.setArg(0, d_totalBgrad);
            kernelAvg.setArg(1, d_gradB[layer]);
            kernelAvg.setArg(2, batchSize);
            kernelAvg.setArg(3, bgradients[layer].size());
            kernelAvg.setArg(4, bgradients[layer][0].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, globalWeightGrad, local_1d));
        }

        // first layer
        int inHeight = input.size();
        int inWidth = input[0].size();
        int firstLayerRows = actBatch[0][0].size();
        int firstLayerCols = actBatch[0][0][0].size();
        cl::NDRange globalWeightGradFirst = calculate_global_1d(WORKSIZE_1D, cweights[0].size() * cweights[0][0].size());
        cl::NDRange globalOutGrad = calculate_global_1d(WORKSIZE_1D, actBatch[0][0].size());
        int row = cgradients[0].size();
        int col = cgradients[0][0].size();
        int totalElements = row * col;
        cl::Buffer d_totalCgrad, d_totalBgrad;
        d_totalCgrad = cl::Buffer(clContext, CL_MEM_READ_WRITE, totalElements * batchSize * sizeof(float));
        d_totalBgrad = cl::Buffer(clContext, CL_MEM_READ_WRITE, totalElements * batchSize * sizeof(float));

        for(int i = 0; i < batchSize; i++) {
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
            CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_gradC[0], d_totalCgrad, 0, i * totalElements * sizeof(float), sizeof(float) * totalElements));

            kernelMatMul.setArg(0, d_input_p_T);
            kernelMatMul.setArg(1, d_incoming[0]);
            kernelMatMul.setArg(2, d_gradC[0]);
            kernelMatMul.setArg(3, inWidth);
            kernelMatMul.setArg(4, inHeight);
            kernelMatMul.setArg(5, firstLayerCols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, calculate_global_2d(size2d, inWidth, firstLayerCols), local_2d));
            kernelScale.setArg(0, d_gradC[0]);
            kernelScale.setArg(1, d_gradC[0]);
            kernelScale.setArg(2, alpha);
            kernelScale.setArg(3, (int)(inWidth * firstLayerCols));
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, calculate_global_1d(WORKSIZE_1D, inWidth * firstLayerCols), local_1d));
            CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_gradB[0], d_totalBgrad, 0, i * totalElements * sizeof(float), sizeof(float) * totalElements));
        }

        // average the gradients
        kernelAvg.setArg(0, d_totalCgrad);
        kernelAvg.setArg(1, d_gradC[0]);
        kernelAvg.setArg(2, batchSize);
        kernelAvg.setArg(3, cgradients[0].size());
        kernelAvg.setArg(4, cgradients[0][0].size());
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, globalWeightGradFirst, local_1d));
        kernelAvg.setArg(0, d_totalBgrad);
        kernelAvg.setArg(1, d_gradB[0]);
        kernelAvg.setArg(2, batchSize);
        kernelAvg.setArg(3, bgradients[0].size());
        kernelAvg.setArg(4, bgradients[0][0].size());
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, globalWeightGradFirst, local_1d));

        // Read the updated weights back to host
        for (int i = 0; i < this->layers; ++i) {
            size_t cweight_size = cweights[i].size() * cweights[i][0].size();
            size_t bweight_size = bweights[i].size() * bweights[i][0].size();
            std::vector<float> takeIn(cweight_size, 0.0f);
            std::vector<float> takeIn2(bweight_size, 0.0f);
            cl::NDRange globalWeightGrad = calculate_global_1d(WORKSIZE_1D, cweight_size);

            size_t prev_layer_size = (i == 0) ? input.size() : actBatch[0][i - 1].size();

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
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_cweights[i], CL_TRUE, 0, sizeof(float) * cweight_size, (void*)takeIn.data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_bweights[i], CL_TRUE, 0, sizeof(float) * bweight_size, (void*)takeIn2.data()));
            cweights[i] = reshape(takeIn, cweights[i].size(), cweights[i][0].size());
            bweights[i] = reshape(takeIn2, bweights[i].size(), bweights[i][0].size());
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_cweights[i], CL_TRUE, 0, sizeof(float) * cweight_size, (void*)takeIn.data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_bweights[i], CL_TRUE, 0, sizeof(float) * bweight_size, (void*)takeIn2.data()));
            cgradients[i] = reshape(takeIn, cweights[i].size(), cweights[i][0].size());
            bgradients[i] = reshape(takeIn2, bweights[i].size(), bweights[i][0].size());
        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn2d::clBackprop (batch): ") + e.what());
    }
}

#endif