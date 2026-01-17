#ifdef USE_CL
#include "mnn1d.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cstdlib>

/**
 * @brief trains the mnn network on a batch of data for single cycle using OpenCL.
 * @param inputs A vector of input vectors.
 * @param targets A vector of target vectors.
 * @param useBuffer Unused for OpenCL implementation.
 */
void mnn1d::clTrainBatch1c(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, bool useBuffer) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs and targets in batch must be the same.");
    }
    if (inputs.empty()) {
        return; // Nothing to train
    }
    if (inputs[0].size() != inSize || targets[0].size() != outSize) {
        throw std::invalid_argument("Input or target dimensions do not match network configuration.");
    }
 
    this->batchSize = inputs.size();

    // Resize batch vectors
    if (dotBatch.size() != layers) {
        dotBatch.resize(layers);
        actBatch.resize(layers);
    }
    for (int i = 0; i < layers; ++i) {
        if (dotBatch[i].size() != batchSize) {
            dotBatch[i].resize(batchSize);
            actBatch[i].resize(batchSize);
            for (int j = 0; j < batchSize; ++j) {
                dotBatch[i][j].resize(width[i]);
                actBatch[i][j].resize(width[i]);
            }
        }
    }
    if (outputBatch.size() != batchSize) {
        outputBatch.resize(batchSize);
        targetBatch.resize(batchSize);
        for(int i=0; i<batchSize; ++i) outputBatch[i].resize(outSize);
        for(int i=0; i<batchSize; ++i) targetBatch[i].resize(outSize);
    }

    std::vector<int> correct(input.size(), -1);
    for (size_t i = 0; i < inputs.size(); ++i) {
        this->inputBatch[i] = softmax(inputs[i]);
    }

    float total_loss = 0.0f;
    if (useBuffer == 0) {
        clForprop(inputBatch);

        int correct_predictions = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Use outputBatch for checking accuracy to avoid re-computation
            if (maxIndex(outputBatch[i]) == maxIndex(targets[i])) {
                correct_predictions++;
                correct[i] = 1;
            }
            else
                correct[i] = 0;
        }

        if (correct_predictions == inputs.size()) {
            currloss = static_cast<float>(total_loss / batchSize);
            std::cout << "All " << inputs.size() << " outputs in the batch are correct. Training complete with error " << currloss << "." << std::endl;
        }
        else {
            std::cout << "Correct Predictions (" << correct_predictions << "/" << inputs.size() << "): ";
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::cout << correct[i] << " ";
            }
            // loss calculation
            for (size_t i = 0; i < inputs.size(); ++i) {
                total_loss += crossEntropy(outputBatch[i], targets[i]);
            }
            currloss = static_cast<float>(total_loss / BATCH_SIZE);
            std::cout << "-> Predictions: " << correct_predictions << "/" << inputs.size() 
                        << "\tAvg. CE Loss: " << currloss << std::endl;
            clBackprop(targets);
        }
    }
    else {
        cl_int err;
        cl::NDRange local_1d(WORKSIZE_1D);
        cl::NDRange local_2d(WORKSIZE_2DX, WORKSIZE_2DY);
        size_t size2d[2] = {WORKSIZE_2DX, WORKSIZE_2DY};

        // --- Buffer Allocation ---
        cl::Buffer d_input_batch, d_target_batch;
        std::vector<cl::Buffer> d_cweights(layers), d_bweights(layers), d_gradC(layers), d_gradB(layers);
        std::vector<cl::Buffer> d_dotProds(layers), d_activate(layers);
        cl::Buffer d_totalCgrad_buf, d_totalBgrad_buf;
        cl::Buffer d_ones, d_preoutgoing_l, d_outgoing_l, d_dpow_l, d_dact_l;

        // Flatten inputs and targets for single transfers
        std::vector<float> flat_inputs, flat_targets;
        for (const auto& vec : inputs) flat_inputs.insert(flat_inputs.end(), vec.begin(), vec.end());
        for (const auto& vec : targets) flat_targets.insert(flat_targets.end(), vec.begin(), vec.end());

        d_input_batch = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, flat_inputs.size() * sizeof(float), (void*)flat_inputs.data(), &err); CL_CHECK(err);
        d_target_batch = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, flat_targets.size() * sizeof(float), (void*)flat_targets.data(), &err); CL_CHECK(err);

        size_t max_total_grad_size = 0;
        size_t max_layer_width = 0;
        for (int i = 0; i < layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            max_total_grad_size = std::max(max_total_grad_size, c_size);
            max_layer_width = std::max(max_layer_width, (size_t)width[i]);

            d_cweights[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, c_size * sizeof(float)); CL_CHECK(err);
            d_bweights[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, b_size * sizeof(float)); CL_CHECK(err);
            d_gradC[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, c_size * sizeof(float)); CL_CHECK(err);
            d_gradB[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, b_size * sizeof(float)); CL_CHECK(err);

            size_t layer_output_size = batchSize * width[i];
            d_dotProds[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, layer_output_size * sizeof(float)); CL_CHECK(err);
            d_activate[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, layer_output_size * sizeof(float)); CL_CHECK(err);
        }
        d_totalCgrad_buf = cl::Buffer(clContext, CL_MEM_READ_WRITE, max_total_grad_size * batchSize * sizeof(float)); CL_CHECK(err);
        d_totalBgrad_buf = cl::Buffer(clContext, CL_MEM_READ_WRITE, max_total_grad_size * batchSize * sizeof(float)); CL_CHECK(err);

        max_layer_width = std::max(max_layer_width, (size_t)inputs[0].size());
        std::vector<float> v1(max_layer_width, 1.0f);
        d_ones = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * max_layer_width, v1.data(), &err); CL_CHECK(err);
        d_preoutgoing_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width); CL_CHECK(err);
        d_outgoing_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width); CL_CHECK(err);
        d_dpow_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width); CL_CHECK(err);
        d_dact_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width); CL_CHECK(err);

        for (int i = 0; i < layers; ++i) {
            std::vector<float> flat_c = flatten(cweights[i]);
            std::vector<float> flat_b = flatten(bweights[i]);
            CL_CHECK(clCommandQueue.enqueueWriteBuffer(d_cweights[i], CL_TRUE, 0, flat_c.size() * sizeof(float), flat_c.data()));
            CL_CHECK(clCommandQueue.enqueueWriteBuffer(d_bweights[i], CL_TRUE, 0, flat_b.size() * sizeof(float), flat_b.data()));
        }

        // --- Forward Propagation (adapted from mnn1d::clForprop(batch)) ---
        cl::Buffer d_current_act = d_input_batch;
        cl::Kernel kernelForwardBatch = kernels.at("kernelLayerForwardBatch2");
        cl::Kernel kernelSigmoid = kernels.at("sigmoid");

        // First layer
        int single_input_size = inputs[0].size();
        kernelForwardBatch.setArg(0, d_current_act);
        kernelForwardBatch.setArg(1, d_dotProds[0]);
        kernelForwardBatch.setArg(2, d_cweights[0]);
        kernelForwardBatch.setArg(3, d_bweights[0]);
        kernelForwardBatch.setArg(4, (int)batchSize);
        kernelForwardBatch.setArg(5, (int)single_input_size);
        kernelForwardBatch.setArg(6, (int)width[0]);
        kernelForwardBatch.setArg(7, order);
        cl::NDRange globalForward(batchSize, width[0]);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForwardBatch, cl::NullRange, globalForward, cl::NullRange));

        kernelSigmoid.setArg(0, d_dotProds[0]);
        kernelSigmoid.setArg(1, d_activate[0]);
        kernelSigmoid.setArg(2, (int)(batchSize * width[0]));
        cl::NDRange globalSig = calculate_global_1d(WORKSIZE_1D, batchSize * width[0]);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSigmoid, cl::NullRange, globalSig, local_1d));

        // Subsequent layers
        for (int i = 1; i < layers; ++i) {
            d_current_act = d_activate[i - 1];
            kernelForwardBatch.setArg(0, d_current_act);
            kernelForwardBatch.setArg(1, d_dotProds[i]);
            kernelForwardBatch.setArg(2, d_cweights[i]);
            kernelForwardBatch.setArg(3, d_bweights[i]);
            kernelForwardBatch.setArg(4, (int)batchSize);
            kernelForwardBatch.setArg(5, (int)width[i - 1]);
            kernelForwardBatch.setArg(6, (int)width[i]);
            kernelForwardBatch.setArg(7, order);
            globalForward = cl::NDRange(batchSize, width[i]);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForwardBatch, cl::NullRange, globalForward, cl::NullRange));

            kernelSigmoid.setArg(0, d_dotProds[i]);
            kernelSigmoid.setArg(1, d_activate[i]);
            kernelSigmoid.setArg(2, (int)(batchSize * width[i]));
            globalSig = calculate_global_1d(WORKSIZE_1D, batchSize * width[i]);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSigmoid, cl::NullRange, globalSig, local_1d));
        }
        CL_CHECK(clCommandQueue.finish());

        // Read the final activation and other results back to the host
        std::vector<float> final_activations(batchSize * outSize);
        CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[layers - 1], CL_TRUE, 0, sizeof(float) * final_activations.size(), final_activations.data()));

        for (int i = 0; i < batchSize; ++i) {
            std::copy(final_activations.begin() + i * outSize, final_activations.begin() + (i + 1) * outSize, outputBatch[i].begin());
            outputBatch[i] = softmax(outputBatch[i]);
        }

        int correct_predictions = 0;
        float total_loss = 0.0f;
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (maxIndex(outputBatch[i]) == maxIndex(targets[i])) {
                correct_predictions++;
                correct[i] = 1;
            }
            else 
                correct[i] = 0;
            total_loss += crossEntropy(outputBatch[i], targets[i]);
        }
        currloss = total_loss / batchSize;

        if (correct_predictions == inputs.size()) {
            currloss = static_cast<float>(total_loss / batchSize);
            std::cout << "All " << inputs.size() << " outputs in the batch are correct. Training complete with error " << currloss << "." << std::endl;
        }
        else {
            // std::cout << "Correct Predictions (" << correct_predictions << "/" << inputs.size() << "): ";
            // for (size_t i = 0; i < inputs.size(); ++i) {
            //     std::cout << correct[i] << " ";
            // }
            // loss calculation
            for (size_t i = 0; i < inputs.size(); ++i) {
                total_loss += crossEntropy(outputBatch[i], targets[i]);
            }
            currloss = static_cast<float>(total_loss / BATCH_SIZE);
            // std::cout << "\n-> Predictions: " << correct_predictions << "/" << inputs.size() << "\tAvg. CE Loss: " << currloss << std::endl;

            // --- Backward Propagation (adapted from mnn1d::clBackprop(batch)) ---
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
            cl::Kernel kernelAvg = kernels.at("matrix_vector_average");

            std::vector<cl::Buffer> d_err_per_batch(batchSize);
            std::vector<std::vector<cl::Buffer>> d_incoming_per_batch(layers);
            std::vector<std::vector<cl::Buffer>> d_dotProds_per_batch(layers);
            std::vector<std::vector<cl::Buffer>> d_activate_per_batch(layers);

            // Read dotProds and activations from device to host for backprop
            for(int i=0; i<layers; ++i) {
                std::vector<float> flat_dots(batchSize * width[i]), flat_acts(batchSize * width[i]);
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_dotProds[i], CL_TRUE, 0, sizeof(float) * flat_dots.size(), flat_dots.data()));
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[i], CL_TRUE, 0, sizeof(float) * flat_acts.size(), flat_acts.data()));
                for(int j=0; j<batchSize; ++j) {
                    std::copy(flat_dots.begin() + j * width[i], flat_dots.begin() + (j+1) * width[i], dotBatch[i][j].begin());
                    std::copy(flat_acts.begin() + j * width[i], flat_acts.begin() + (j+1) * width[i], actBatch[i][j].begin());
                }
            }

            for(int i = 0; i < layers; ++i) {
                d_incoming_per_batch[i].resize(batchSize);
                d_dotProds_per_batch[i].resize(batchSize);
                d_activate_per_batch[i].resize(batchSize);
                for(int j = 0; j < batchSize; ++j) {
                    size_t act_size = width[i];
                    d_incoming_per_batch[i][j] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
                    d_dotProds_per_batch[i][j] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, act_size * sizeof(float), (void*)dotBatch[i][j].data(), &err); CL_CHECK(err);
                    d_activate_per_batch[i][j] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, act_size * sizeof(float), (void*)actBatch[i][j].data(), &err); CL_CHECK(err);
                }
            }

            // Calculate initial error (activate[layers - 1] - expected)
            for (int i = 0; i < batchSize; ++i) {
                d_err_per_batch[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, targets[i].size() * sizeof(float)); CL_CHECK(err);
                cl::Buffer d_out_single(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, actBatch[layers-1][i].size() * sizeof(float), (void*)actBatch[i].data(), &err); CL_CHECK(err);
                cl::Buffer d_exp_single(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, targets[i].size() * sizeof(float), (void*)targets[i].data(), &err); CL_CHECK(err);

                kernelSub.setArg(0, d_out_single);
                kernelSub.setArg(1, d_exp_single);
                kernelSub.setArg(2, d_err_per_batch[i]);
                kernelSub.setArg(3, (int)targets[i].size());
                cl::NDRange globalSub = calculate_global_1d(WORKSIZE_1D, targets[i].size());
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSub, cl::NullRange, globalSub, local_1d));
                CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_err_per_batch[i], d_incoming_per_batch[layers - 1][i], 0, 0, sizeof(float) * targets[i].size()));
            }
            CL_CHECK(clCommandQueue.finish());

            for (int layer = layers - 1; layer >= 1; --layer) {
                int cweight_rows = width[layer - 1];
                int cweight_cols = width[layer];
                size_t cweight_flat_size = cweight_rows * cweight_cols;
                cl::NDRange globalWeightGrad = calculate_global_1d(WORKSIZE_1D, cweight_flat_size);
                cl::NDRange globalOutGrad = calculate_global_1d(WORKSIZE_1D, cweight_rows);

                for (int i = 0; i < batchSize; ++i) {
                    kernelVecxVec2Mat.setArg(0, d_activate_per_batch[layer - 1][i]);
                    kernelVecxVec2Mat.setArg(1, d_incoming_per_batch[layer][i]);
                    kernelVecxVec2Mat.setArg(2, d_gradC[layer]);
                    kernelVecxVec2Mat.setArg(3, cweight_rows);
                    kernelVecxVec2Mat.setArg(4, cweight_cols);
                    CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelVecxVec2Mat, cl::NullRange, globalWeightGrad, local_1d));
                    CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_gradC[layer], d_totalCgrad_buf, 0, i * cweight_flat_size * sizeof(float), sizeof(float) * cweight_flat_size));

                    kernelVecxVec2Mat.setArg(0, d_ones);
                    kernelVecxVec2Mat.setArg(1, d_incoming_per_batch[layer][i]);
                    kernelVecxVec2Mat.setArg(2, d_gradB[layer]);
                    kernelVecxVec2Mat.setArg(3, cweight_rows);
                    kernelVecxVec2Mat.setArg(4, cweight_cols);
                    CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelVecxVec2Mat, cl::NullRange, globalWeightGrad, local_1d));
                    CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_gradB[layer], d_totalBgrad_buf, 0, i * cweight_flat_size * sizeof(float), sizeof(float) * cweight_flat_size));

                    cl::Buffer d_C_T(clContext, CL_MEM_READ_WRITE, cweight_flat_size * sizeof(float)); CL_CHECK(err);
                    kernelTranspose.setArg(0, d_cweights[layer]);
                    kernelTranspose.setArg(1, d_C_T);
                    kernelTranspose.setArg(2, cweight_rows);
                    kernelTranspose.setArg(3, cweight_cols);
                    cl::NDRange globalTranspose = calculate_global_2d(size2d, cweight_rows, cweight_cols);
                    CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, globalTranspose, local_2d));

                    kernelVecxMat2Vec.setArg(0, d_incoming_per_batch[layer][i]);
                    kernelVecxMat2Vec.setArg(1, d_C_T);
                    kernelVecxMat2Vec.setArg(2, d_preoutgoing_l);
                    kernelVecxMat2Vec.setArg(3, cweight_cols);
                    kernelVecxMat2Vec.setArg(4, cweight_rows);
                    CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelVecxMat2Vec, cl::NullRange, globalOutGrad, local_1d));

                    kernelDPow.setArg(0, d_activate_per_batch[layer - 1][i]);
                    kernelDPow.setArg(1, d_dpow_l);
                    kernelDPow.setArg(2, order);
                    kernelDPow.setArg(3, cweight_rows);
                    CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelDPow, cl::NullRange, globalOutGrad, local_1d));

                    kernelSigmoidDer.setArg(0, d_dotProds_per_batch[layer - 1][i]);
                    kernelSigmoidDer.setArg(1, d_dact_l);
                    kernelSigmoidDer.setArg(2, cweight_rows);
                    CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSigmoidDer, cl::NullRange, globalOutGrad, local_1d));

                    kernelHadamard.setArg(0, d_preoutgoing_l);
                    kernelHadamard.setArg(1, d_dpow_l);
                    kernelHadamard.setArg(2, d_dact_l);
                    kernelHadamard.setArg(3, d_incoming_per_batch[layer - 1][i]);
                    kernelHadamard.setArg(4, 1);
                    kernelHadamard.setArg(5, cweight_rows);
                    CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelHadamard, cl::NullRange, globalOutGrad, local_1d));
                    d_C_T = cl::Buffer();
                }
                CL_CHECK(clCommandQueue.finish());

                int rows = cweights[layer].size();
                int cols = cweights[layer][0].size();
                cl::NDRange globalAvg = calculate_global_2d(size2d, rows, cols);
                kernelAvg.setArg(0, d_totalCgrad_buf);
                kernelAvg.setArg(1, d_gradC[layer]);
                kernelAvg.setArg(2, batchSize);
                kernelAvg.setArg(3, rows);
                kernelAvg.setArg(4, cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, globalAvg, local_2d));

                kernelScale.setArg(0, d_gradC[layer]);
                kernelScale.setArg(1, d_gradC[layer]);
                kernelScale.setArg(2, ALPHA);
                kernelScale.setArg(3, (int)cweight_flat_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGrad, local_1d));

                kernelAvg.setArg(0, d_totalBgrad_buf);
                kernelAvg.setArg(1, d_gradB[layer]);
                kernelAvg.setArg(2, batchSize);
                kernelAvg.setArg(3, rows);
                kernelAvg.setArg(4, cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, globalAvg, local_2d));

                kernelScale.setArg(0, d_gradB[layer]);
                kernelScale.setArg(1, d_gradB[layer]);
                kernelScale.setArg(2, 1.0f - ALPHA);
                kernelScale.setArg(3, (int)cweight_flat_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGrad, local_1d));
            }

            // First layer
            int cweight_rows_first = single_input_size;
            int cweight_cols_first = width[0];
            size_t cweight_flat_size_first = cweight_rows_first * cweight_cols_first;
            cl::NDRange globalWeightGradFirst = calculate_global_1d(WORKSIZE_1D, cweight_flat_size_first);

            for (int i = 0; i < batchSize; ++i) {
                cl::Buffer d_in_single(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputs[i].size() * sizeof(float), (void*)inputs[i].data(), &err); CL_CHECK(err);
                kernelVecxVec2Mat.setArg(0, d_incoming_per_batch[0][i]);
                kernelVecxVec2Mat.setArg(1, d_in_single);
                kernelVecxVec2Mat.setArg(2, d_gradC[0]);
                kernelVecxVec2Mat.setArg(3, cweight_rows_first);
                kernelVecxVec2Mat.setArg(4, cweight_cols_first);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelVecxVec2Mat, cl::NullRange, globalWeightGradFirst, local_1d));
                CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_gradC[0], d_totalCgrad_buf, 0, i * cweight_flat_size_first * sizeof(float), sizeof(float) * cweight_flat_size_first));

                kernelVecxVec2Mat.setArg(0, d_incoming_per_batch[0][i]);
                kernelVecxVec2Mat.setArg(1, d_ones);
                kernelVecxVec2Mat.setArg(2, d_gradB[0]);
                kernelVecxVec2Mat.setArg(3, cweight_rows_first);
                kernelVecxVec2Mat.setArg(4, cweight_cols_first);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelVecxVec2Mat, cl::NullRange, globalWeightGradFirst, local_1d));
                CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_gradB[0], d_totalBgrad_buf, 0, i * cweight_flat_size_first * sizeof(float), sizeof(float) * cweight_flat_size_first));
                d_in_single = cl::Buffer();
            }
            CL_CHECK(clCommandQueue.finish());

            int rows_first = cweights[0].size();
            int cols_first = cweights[0][0].size();
            cl::NDRange globalAvgFirst = calculate_global_2d(size2d, rows_first, cols_first);
            kernelAvg.setArg(0, d_totalCgrad_buf);
            kernelAvg.setArg(1, d_gradC[0]);
            kernelAvg.setArg(2, batchSize);
            kernelAvg.setArg(3, rows_first);
            kernelAvg.setArg(4, cols_first);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, globalAvgFirst, local_2d));

            kernelScale.setArg(0, d_gradC[0]);
            kernelScale.setArg(1, d_gradC[0]);
            kernelScale.setArg(2, ALPHA);
            kernelScale.setArg(3, (int)cweight_flat_size_first);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGradFirst, local_1d));

            kernelAvg.setArg(0, d_totalBgrad_buf);
            kernelAvg.setArg(1, d_gradB[0]);
            kernelAvg.setArg(2, batchSize);
            kernelAvg.setArg(3, rows_first);
            kernelAvg.setArg(4, cols_first);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, globalAvgFirst, local_2d));

            kernelScale.setArg(0, d_gradB[0]);
            kernelScale.setArg(1, d_gradB[0]);
            kernelScale.setArg(2, 1.0f - ALPHA);
            kernelScale.setArg(3, (int)cweight_flat_size_first);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGradFirst, local_1d));
            CL_CHECK(clCommandQueue.finish());

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
            CL_CHECK(clCommandQueue.finish());

            for(int i = 0; i < batchSize; ++i) {
                d_err_per_batch[i] = cl::Buffer();
            }
            for(int i = 0; i < layers; ++i) {
                for(int j = 0; j < batchSize; ++j) {
                    d_incoming_per_batch[i][j] = cl::Buffer();
                    d_dotProds_per_batch[i][j] = cl::Buffer();
                    d_activate_per_batch[i][j] = cl::Buffer();
                }
            }
        }
        
        // --- Buffer Cleanup ---
        d_input_batch = cl::Buffer(); d_target_batch = cl::Buffer();
        d_totalCgrad_buf = cl::Buffer(); d_totalBgrad_buf = cl::Buffer();
        d_ones = cl::Buffer(); d_preoutgoing_l = cl::Buffer(); d_outgoing_l = cl::Buffer(); d_dpow_l = cl::Buffer(); d_dact_l = cl::Buffer();
        for (int i = 0; i < layers; ++i) {
            d_cweights[i] = cl::Buffer(); d_bweights[i] = cl::Buffer();
            d_gradC[i] = cl::Buffer(); d_gradB[i] = cl::Buffer();
            d_dotProds[i] = cl::Buffer(); d_activate[i] = cl::Buffer();
        }
    }
}

#endif // USE_CL