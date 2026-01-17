#ifdef USE_CL
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include "mnn1d.hpp"
#include "mnn2d.hpp"

/**
 * @brief batch backpropgation using opencl for mnn
 * @param expected expected result from forprop
 */
void mnn1d::clBackprop(const std::vector<std::vector<float>>& expected) {
    try {
        cl_int err;
        int batchSize = expected.size();
        if (batchSize == 0) {
            throw std::runtime_error("Batch size should be greater than 0.");
        }
        cl::NDRange local_1d(WORKSIZE_1D);
        cl::NDRange local_2d(WORKSIZE_2DX, WORKSIZE_2DY);
        size_t size2d[2] = {WORKSIZE_2DX, WORKSIZE_2DY};

        size_t inputSize = inputBatch[0].size();
        size_t outputSize = output.size();

        std::vector<cl::Buffer> d_in(batchSize), d_exp(batchSize), d_out(batchSize), d_err(batchSize);
        std::vector<std::vector<cl::Buffer>> d_incoming(this->layers), d_dotProds(this->layers), d_activate(this->layers);
        std::vector<cl::Buffer> d_cweights(this->layers);
        std::vector<cl::Buffer> d_bweights(this->layers);
        std::vector<cl::Buffer> d_gradC(this->layers);
        std::vector<cl::Buffer> d_gradB(this->layers);
        cl::Buffer d_ones;
        cl::Buffer d_preoutgoing_l, d_outgoing_l, d_dpow_l, d_dact_l;

        cl::Kernel kernelSigmoidDer, kernelSub, kernelDPow, kernelvxv2m, kernelvxv2v, kernelUpdateWeights, 
                kernelHadamard, kernelTranspose, kernelAvg, kernelScale;
        kernelSigmoidDer = kernels.at("sigmoidDer");
        kernelSub = kernels.at("subtract");
        kernelDPow = kernels.at("dPower");
        kernelvxv2m = kernels.at("vecxvec2mat"); // outer product
        kernelvxv2v = kernels.at("vecxmat2vec"); // vec x mat
        kernelTranspose = kernels.at("transpose");
        kernelHadamard = kernels.at("hadamard2");
        kernelDPow = kernels.at("dPower");
        kernelAvg = kernels.at("matrix_vector_average");
        kernelScale = kernels.at("scaleByValue");
        
        switch (weightUpdateType) {
            case 0: kernelUpdateWeights = kernels.at("kernelUpdateWeights"); break;
            case 1: kernelUpdateWeights = kernels.at("kernelUpdateWeightsWithL1"); break;
            case 2: kernelUpdateWeights = kernels.at("kernelUpdateWeightsWithL2"); break;
            case 3: kernelUpdateWeights = kernels.at("kernelUpdateWeightsElasticNet"); break;
            case 4: kernelUpdateWeights = kernels.at("kernelUpdateWeightsWithWeightDecay"); break;
            case 5: kernelUpdateWeights = kernels.at("kernelUpdateWeightsDropout"); break;
            default: throw std::runtime_error("Invalid weight update type");
        }

        // buffer data and space allotment
        // Get device from context
        std::vector<cl::Device> devices = clContext.getInfo<CL_CONTEXT_DEVICES>();
        if (devices.empty()) throw std::runtime_error("No devices found in context.");
        this->device = devices[0];

        std::vector<cl::CommandQueue> clStream(batchSize);
        for (int i = 0; i < batchSize; ++i) {
            clStream[i] = cl::CommandQueue(clContext, device, 0, &err); CL_CHECK(err);
        }
        for (int i = 0; i < batchSize; ++i) {
            d_in[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputSize, inputBatch[i].data(), &err); CL_CHECK(err);
            d_exp[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * expected[i].size(), (void*)expected[i].data(), &err); CL_CHECK(err);
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
            d_activate[i].resize(batchSize);
            d_dotProds[i].resize(batchSize);
            d_incoming[i].resize(batchSize);
            for(int j = 0; j < batchSize; j++) {
                d_activate[i][j] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * actBatch[i][j].size(), (void*)actBatch[i][j].data(), &err); CL_CHECK(err);
                d_dotProds[i][j] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * dotBatch[i][j].size(), (void*)dotBatch[i][j].data(), &err); CL_CHECK(err);
                d_incoming[i][j] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * actBatch[i][j].size()); CL_CHECK(err);
            }
        }

        size_t max_layer_width = 0;
        for(int w : width) max_layer_width = std::max(max_layer_width, (size_t)w);
        max_layer_width = std::max(max_layer_width, inputSize);
        d_preoutgoing_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width);
        d_outgoing_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width);
        d_dpow_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width);
        d_dact_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width);
        std::vector<float> v1(max_layer_width, 1.0f);
        d_ones = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * v1.size(), v1.data(), &err); CL_CHECK(err);

        for(int i = 0; i < batchSize; ++i) {
            // Calculate initial error (output - expected)
            kernelSub.setArg(0, d_out[i]);
            kernelSub.setArg(1, d_exp[i]);
            kernelSub.setArg(2, d_err[i]);
            kernelSub.setArg(3, (int)outputSize);
            cl::NDRange globalSub = calculate_global_1d(WORKSIZE_1D, outputSize);
            CL_CHECK(clStream[i].enqueueNDRangeKernel(kernelSub, cl::NullRange, globalSub, local_1d));
            CL_CHECK(clStream[i].enqueueCopyBuffer(d_err[i], d_incoming[layers - 1][i], 0, 0, sizeof(float) * outputSize));
        }
    
        // Synchronize all streams to ensure initial error calculation is complete for all batch items
        for (int i = 0; i < batchSize; ++i) {
            clStream[i].finish();
        }

        for(int layer = layers - 1; layer >= 1; --layer) {
            int cweight_rows = cweights[layer].size();
            int cweight_cols = cweights[layer][0].size();
            int prev_size = cweight_rows;
            size_t cweight_flat_size = cweight_rows * cweight_cols;

            cl::NDRange globalWeightGrad = calculate_global_1d(WORKSIZE_1D, cweight_flat_size);
            cl::NDRange globalOutGrad = calculate_global_1d(WORKSIZE_1D, prev_size);

            int row = cgradients[layer].size();
            int col = cgradients[layer][0].size();
            int totalElements = row * col;
            cl::Buffer d_totalCgrad, d_totalBgrad;
            d_totalCgrad = cl::Buffer(clContext, CL_MEM_READ_WRITE, totalElements * batchSize * sizeof(float));
            d_totalBgrad = cl::Buffer(clContext, CL_MEM_READ_WRITE, totalElements * batchSize * sizeof(float));

            for(int i = 0; i < batchSize; ++i) {
                // dL/dC_l (Outer Product)
                kernelvxv2m.setArg(0, d_incoming[layer][i]);
                kernelvxv2m.setArg(1, d_activate[layer - 1][i]);
                kernelvxv2m.setArg(2, d_gradC[layer]); kernelvxv2m.setArg(3, cweight_rows); kernelvxv2m.setArg(4, cweight_cols);
                CL_CHECK(clStream[i].enqueueNDRangeKernel(kernelvxv2m, cl::NullRange, globalWeightGrad, local_1d));
                CL_CHECK(clStream[i].enqueueCopyBuffer(d_gradC[layer], d_totalCgrad, 0, i * totalElements * sizeof(float), sizeof(float) * totalElements));

                // dL/dB_l (Outer Product)
                kernelvxv2m.setArg(0, d_incoming[layer][i]);
                kernelvxv2m.setArg(1, d_ones);
                kernelvxv2m.setArg(2, d_gradB[layer]);
                kernelvxv2m.setArg(3, cweight_rows); kernelvxv2m.setArg(4, cweight_cols);
                CL_CHECK(clStream[i].enqueueNDRangeKernel(kernelvxv2m, cl::NullRange, globalWeightGrad, local_1d));
                CL_CHECK(clStream[i].enqueueCopyBuffer(d_gradB[layer], d_totalBgrad, 0, i * totalElements * sizeof(float), sizeof(float) * totalElements));

                // --- Outgoing Gradient Calculation ---
                cl::Buffer d_C_T(clContext, CL_MEM_READ_WRITE, cweight_flat_size * sizeof(float));
                kernelTranspose.setArg(0, d_cweights[layer]);
                kernelTranspose.setArg(1, d_C_T);
                kernelTranspose.setArg(2, cweight_rows);
                kernelTranspose.setArg(3, cweight_cols);
                cl::NDRange globalTranspose = calculate_global_2d(size2d, cweight_rows, cweight_cols);
                CL_CHECK(clStream[i].enqueueNDRangeKernel(kernelTranspose, cl::NullRange, globalTranspose, local_2d));

                // incoming gradient x C^T
                kernelvxv2v.setArg(0, d_incoming[layer][i]);
                kernelvxv2v.setArg(1, d_C_T);
                kernelvxv2v.setArg(2, d_preoutgoing_l);
                kernelvxv2v.setArg(3, cweight_cols); // matRows
                kernelvxv2v.setArg(4, cweight_rows); // matCols
                CL_CHECK(clStream[i].enqueueNDRangeKernel(kernelvxv2v, cl::NullRange, globalOutGrad, local_1d));

                // derivative of power
                kernelDPow.setArg(0, d_activate[layer - 1][i]);
                kernelDPow.setArg(1, d_dpow_l);
                kernelDPow.setArg(2, order);
                kernelDPow.setArg(3, prev_size);
                CL_CHECK(clStream[i].enqueueNDRangeKernel(kernelDPow, cl::NullRange, globalOutGrad, local_1d));

                // derivative of activation
                kernelSigmoidDer.setArg(0, d_dotProds[layer - 1][i]);
                kernelSigmoidDer.setArg(1, d_dact_l);
                kernelSigmoidDer.setArg(2, prev_size);
                CL_CHECK(clStream[i].enqueueNDRangeKernel(kernelSigmoidDer, cl::NullRange, globalOutGrad, local_1d));

                // outgoing gradient = (dl/dz_l x C^T) . dprev_p . dprevAct
                kernelHadamard.setArg(0, d_preoutgoing_l);
                kernelHadamard.setArg(1, d_dpow_l);
                kernelHadamard.setArg(2, d_dact_l);
                kernelHadamard.setArg(3, d_incoming[layer - 1][i]);
                kernelHadamard.setArg(4, 1);
                kernelHadamard.setArg(5, prev_size);
                CL_CHECK(clStream[i].enqueueNDRangeKernel(kernelHadamard, cl::NullRange, globalOutGrad, local_1d));
            }

            // average the gradients
            cl::NDRange globalAvg = calculate_global_2d(size2d, row, col);
            kernelAvg.setArg(0, d_totalCgrad);
            kernelAvg.setArg(1, d_gradC[layer]);
            kernelAvg.setArg(2, batchSize);
            kernelAvg.setArg(3, row);
            kernelAvg.setArg(4, col);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, globalAvg, local_2d));

            // scale gradc by ALPHA
            kernelScale.setArg(0, d_gradC[layer]);
            kernelScale.setArg(1, d_gradC[layer]);
            kernelScale.setArg(2, ALPHA);
            kernelScale.setArg(3, totalElements);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGrad, local_1d));

            kernelAvg.setArg(0, d_totalBgrad);
            kernelAvg.setArg(1, d_gradB[layer]);
            kernelAvg.setArg(2, batchSize);
            kernelAvg.setArg(3, row);
            kernelAvg.setArg(4, col);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, globalAvg, local_2d));

            // scale gradb by 1-ALPHA
            kernelScale.setArg(0, d_gradB[layer]);
            kernelScale.setArg(1, d_gradB[layer]);
            kernelScale.setArg(2, 1.0f - ALPHA);
            kernelScale.setArg(3, totalElements);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGrad, local_1d));
        }

        // for first layer
        int cweight_rows_first = inputSize;
        int cweight_cols_first = width[0];
        size_t cweight_flat_size_first = cweight_rows_first * cweight_cols_first;
        cl::NDRange globalWeightGradFirst = calculate_global_1d(WORKSIZE_1D, cweight_flat_size_first); // Outer product size

        int row = cgradients[0].size();
        int col = cgradients[0][0].size();
        int totalElements = row * col;
        cl::Buffer d_totalCgrad, d_totalBgrad;
        d_totalCgrad = cl::Buffer(clContext, CL_MEM_READ_WRITE, totalElements * batchSize * sizeof(float));
        d_totalBgrad = cl::Buffer(clContext, CL_MEM_READ_WRITE, totalElements * batchSize * sizeof(float));

        for(int i = 0; i < batchSize; ++i) {
            // dL/dC_0
            kernelvxv2m.setArg(0, d_incoming[0][i]);
            kernelvxv2m.setArg(1, d_in[i]);
            kernelvxv2m.setArg(2, d_gradC[0]);
            kernelvxv2m.setArg(3, cweight_rows_first); kernelvxv2m.setArg(4, cweight_cols_first); // Correct dimensions for outer product
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2m, cl::NullRange, globalWeightGradFirst, local_1d));
            CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_gradC[0], d_totalCgrad, 0, i * cweight_flat_size_first * sizeof(float), sizeof(float) * cweight_flat_size_first));

            // dL/dB_0
            kernelvxv2m.setArg(0, d_incoming[0][i]);
            kernelvxv2m.setArg(1, d_ones);
            kernelvxv2m.setArg(2, d_gradB[0]);
            kernelvxv2m.setArg(3, cweight_rows_first); kernelvxv2m.setArg(4, cweight_cols_first); // Correct dimensions for outer product
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2m, cl::NullRange, globalWeightGradFirst, local_1d));
            CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_gradB[0], d_totalBgrad, 0, i * cweight_flat_size_first * sizeof(float), sizeof(float) * cweight_flat_size_first));
        }

        // average the gradients
        cl::NDRange globalAvgFirst = calculate_global_2d(size2d, row, col);
        kernelAvg.setArg(0, d_totalCgrad);
        kernelAvg.setArg(1, d_gradC[0]);
        kernelAvg.setArg(2, batchSize);
        kernelAvg.setArg(3, row);
        kernelAvg.setArg(4, col);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, globalAvgFirst, local_2d));

        // scale gradc by ALPHA
        kernelScale.setArg(0, d_gradC[0]);
        kernelScale.setArg(1, d_gradC[0]);
        kernelScale.setArg(2, ALPHA);
        kernelScale.setArg(3, (int)cweight_flat_size_first);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGradFirst, local_1d));

        kernelAvg.setArg(0, d_totalBgrad);
        kernelAvg.setArg(1, d_gradB[0]);
        kernelAvg.setArg(2, batchSize);
        kernelAvg.setArg(3, row);
        kernelAvg.setArg(4, col);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, globalAvgFirst, local_2d));

        // scale gradb by 1-ALPHA
        kernelScale.setArg(0, d_gradB[0]);
        kernelScale.setArg(1, d_gradB[0]);
        kernelScale.setArg(2, 1.0f - ALPHA);
        kernelScale.setArg(3, (int)cweight_flat_size_first);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGradFirst, local_1d));

        // Read the updated weights back to host
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
        throw std::runtime_error(std::string("Exception in mnn1d::clBackprop (batch): ") + e.what());
    }
}

#endif