#ifdef USE_CU
#include "mnn1d.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>

/**
 * @brief Trains the mnn network on a batch of data with optimized buffer management.
 * @param inputs A vector of input vectors.
 * @param targets A vector of target vectors.
 */
void mnn::cuBufTrainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets) {
    if (inputs.size() != targets.size() || inputs.empty()) {
        throw std::invalid_argument("Invalid batch training data.");
    }
    this->batchSize = inputs.size();

    // --- Buffer Allocation ---
    float *d_input_batch = nullptr, *d_target_batch = nullptr;
    std::vector<float*> d_cweights(layers), d_bweights(layers), d_gradC(layers), d_gradB(layers);
    std::vector<float*> d_dotProds(layers), d_activate(layers);
    float *d_totalCgrad = nullptr, *d_totalBgrad = nullptr;
    float *d_ones = nullptr, *d_preoutgoing_l = nullptr, *d_dpow_l = nullptr, *d_dact_l = nullptr;
    dim3 block_1d(WORKSIZE_1D);
    dim3 block_2d(WORKSIZE_2D_X, WORKSIZE_2D_Y);


    try {
        // Flatten inputs and targets for single transfers
        std::vector<float> flat_inputs, flat_targets;
        for(const auto& vec : inputs) flat_inputs.insert(flat_inputs.end(), vec.begin(), vec.end());
        for(const auto& vec : targets) flat_targets.insert(flat_targets.end(), vec.begin(), vec.end());

        CU_CHECK(cudaMalloc(&d_input_batch, flat_inputs.size() * sizeof(float)));
        CU_CHECK(cudaMemcpy(d_input_batch, flat_inputs.data(), flat_inputs.size() * sizeof(float), cudaMemcpyHostToDevice));
        CU_CHECK(cudaMalloc(&d_target_batch, flat_targets.size() * sizeof(float)));
        CU_CHECK(cudaMemcpy(d_target_batch, flat_targets.data(), flat_targets.size() * sizeof(float), cudaMemcpyHostToDevice));

        size_t max_total_grad_size = 0;
        for (int i = 0; i < layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            size_t layer_output_size = batchSize * width[i];
            max_total_grad_size = std::max(max_total_grad_size, c_size);

            CU_CHECK(cudaMalloc(&d_cweights[i], c_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_bweights[i], b_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_gradC[i], c_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_gradB[i], b_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_dotProds[i], layer_output_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_activate[i], layer_output_size * sizeof(float)));
        }
        CU_CHECK(cudaMalloc(&d_totalCgrad, max_total_grad_size * batchSize * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_totalBgrad, max_total_grad_size * batchSize * sizeof(float)));

        int max_layer_width = 0;
        for(int w : width) max_layer_width = std::max(max_layer_width, w);
        max_layer_width = std::max(max_layer_width, (int)inputs[0].size());
        std::vector<float> v1(max_layer_width, 1.0f);
        CU_CHECK(cudaMalloc(&d_ones, sizeof(float) * max_layer_width));
        CU_CHECK(cudaMemcpy(d_ones, v1.data(), sizeof(float) * max_layer_width, cudaMemcpyHostToDevice));
        CU_CHECK(cudaMalloc(&d_preoutgoing_l, sizeof(float) * max_layer_width));
        CU_CHECK(cudaMalloc(&d_dpow_l, sizeof(float) * max_layer_width));
        CU_CHECK(cudaMalloc(&d_dact_l, sizeof(float) * max_layer_width));


        int totalEpochs = 0;
        if (this->epochs < 1) this->epochs = 100;
        float initialLR = this->learningRate;

        // --- Training Loop ---
        while (true) {
            totalEpochs++;

            for (int i = 0; i < layers; ++i) {
                std::vector<float> flat_c = flatten(cweights[i]);
                std::vector<float> flat_b = flatten(bweights[i]);
                CU_CHECK(cudaMemcpy(d_cweights[i], flat_c.data(), flat_c.size() * sizeof(float), cudaMemcpyHostToDevice));
                CU_CHECK(cudaMemcpy(d_bweights[i], flat_b.data(), flat_b.size() * sizeof(float), cudaMemcpyHostToDevice));
            }

            // --- Forward Propagation ---
            float* d_current_act = d_input_batch;
            int single_input_size = inputs[0].size();

            // First layer
            kernelLayerForwardBatch2<<<dim3(batchSize, width[0]), dim3(1, 1)>>>(
                d_current_act, d_dotProds[0], d_cweights[0], d_bweights[0],
                batchSize, single_input_size, width[0], order
            );
            sigmoid<<<calculate_grid_1d(batchSize * width[0], WORKSIZE_1D), block_1d>>>(
                d_dotProds[0], d_activate[0], batchSize * width[0]
            );

            // Subsequent layers
            for (int i = 1; i < layers; ++i) {
                d_current_act = d_activate[i - 1];
                kernelLayerForwardBatch2<<<dim3(batchSize, width[i]), dim3(1, 1)>>>(
                    d_current_act, d_dotProds[i], d_cweights[i], d_bweights[i],
                    batchSize, width[i-1], width[i], order
                );
                sigmoid<<<calculate_grid_1d(batchSize * width[i], WORKSIZE_1D), block_1d>>>(
                    d_dotProds[i], d_activate[i], batchSize * width[i]
                );
            }
            CU_CHECK(cudaDeviceSynchronize());

            // Read final activations and intermediate data for checking and backprop
            std::vector<float> final_activations(batchSize * outSize);
            CU_CHECK(cudaMemcpy(final_activations.data(), d_activate[layers - 1], sizeof(float) * final_activations.size(), cudaMemcpyDeviceToHost));
            for (int i = 0; i < batchSize; ++i) {
                std::copy(final_activations.begin() + i * outSize, final_activations.begin() + (i + 1) * outSize, outputBatch[i].begin());
            }
            for(int i=0; i<layers; ++i) {
                std::vector<float> flat_dots(batchSize * width[i]), flat_acts(batchSize * width[i]);
                CU_CHECK(cudaMemcpy(flat_dots.data(), d_dotProds[i], sizeof(float) * flat_dots.size(), cudaMemcpyDeviceToHost));
                CU_CHECK(cudaMemcpy(flat_acts.data(), d_activate[i], sizeof(float) * flat_acts.size(), cudaMemcpyDeviceToHost));
                for(int j=0; j<batchSize; ++j) {
                    std::copy(flat_dots.begin() + j*width[i], flat_dots.begin() + (j+1)*width[i], dotBatch[i][j].begin());
                    std::copy(flat_acts.begin() + j*width[i], flat_acts.begin() + (j+1)*width[i], actBatch[i][j].begin());
                }
            }

            int correct_predictions = 0;
            float total_loss = 0.0f;
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (maxIndex(outputBatch[i]) == maxIndex(targets[i])) correct_predictions++;
                total_loss += crossEntropy(outputBatch[i], targets[i]);
            }
            currloss = total_loss / batchSize;

            if (correct_predictions == inputs.size()) {
                std::cout << "All " << inputs.size() << " outputs correct after " << totalEpochs << " epochs. Loss: " << currloss << std::endl;
                break;
            } else {
                std::cout << "-> Epoch: " << totalEpochs << "\tPredictions: " << correct_predictions << "/" << inputs.size()
                          << "\tAvg. CE Loss: " << currloss << std::endl;
            }

            if (totalEpochs >= this->epochs) {
                std::cout << "Epoch limit reached. Continuing training." << std::endl;
                this->epochs += 50;
            }

            // --- Backward Propagation ---
            std::vector<float*> d_err_per_batch(batchSize);
            std::vector<std::vector<float*>> d_incoming_per_batch(layers), d_dotProds_per_batch(layers), d_activate_per_batch(layers);
            for(int i=0; i<layers; ++i) {
                d_incoming_per_batch[i].resize(batchSize);
                d_dotProds_per_batch[i].resize(batchSize);
                d_activate_per_batch[i].resize(batchSize);
                for(int j=0; j<batchSize; ++j) {
                    CU_CHECK(cudaMalloc(&d_incoming_per_batch[i][j], width[i] * sizeof(float)));
                    CU_CHECK(cudaMalloc(&d_dotProds_per_batch[i][j], width[i] * sizeof(float)));
                    CU_CHECK(cudaMalloc(&d_activate_per_batch[i][j], width[i] * sizeof(float)));
                    CU_CHECK(cudaMemcpy(d_dotProds_per_batch[i][j], dotBatch[i][j].data(), width[i] * sizeof(float), cudaMemcpyHostToDevice));
                    CU_CHECK(cudaMemcpy(d_activate_per_batch[i][j], actBatch[i][j].data(), width[i] * sizeof(float), cudaMemcpyHostToDevice));
                }
            }

            for (int i = 0; i < batchSize; ++i) {
                CU_CHECK(cudaMalloc(&d_err_per_batch[i], targets[i].size() * sizeof(float)));
                float *d_out_single, *d_exp_single;
                CU_CHECK(cudaMalloc(&d_out_single, outputBatch[i].size() * sizeof(float)));
                CU_CHECK(cudaMalloc(&d_exp_single, targets[i].size() * sizeof(float)));
                CU_CHECK(cudaMemcpy(d_out_single, outputBatch[i].data(), outputBatch[i].size() * sizeof(float), cudaMemcpyHostToDevice));
                CU_CHECK(cudaMemcpy(d_exp_single, targets[i].data(), targets[i].size() * sizeof(float), cudaMemcpyHostToDevice));
                subtract<<<calculate_grid_1d(targets[i].size(), WORKSIZE_1D), block_1d>>>(d_out_single, d_exp_single, d_err_per_batch[i], targets[i].size());
                CU_CHECK(cudaMemcpy(d_incoming_per_batch[layers - 1][i], d_err_per_batch[i], sizeof(float) * targets[i].size(), cudaMemcpyDeviceToDevice));
                cudaFree(d_out_single); cudaFree(d_exp_single);
            }

            for (int layer = layers - 1; layer >= 1; --layer) {
                int cweight_rows = width[layer - 1], cweight_cols = width[layer];
                size_t cweight_flat_size = cweight_rows * cweight_cols;
                dim3 gridWeightGrad = calculate_grid_1d(cweight_flat_size, WORKSIZE_1D);
                dim3 gridOutGrad = calculate_grid_1d(cweight_rows, WORKSIZE_1D);

                for (int i = 0; i < batchSize; ++i) {
                    vecxvec2mat<<<gridWeightGrad, block_1d>>>(d_activate_per_batch[layer - 1][i], d_incoming_per_batch[layer][i], d_gradC[layer], cweight_rows, cweight_cols);
                    CU_CHECK(cudaMemcpy(d_totalCgrad + i * cweight_flat_size, d_gradC[layer], sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));
                    vecxvec2mat<<<gridWeightGrad, block_1d>>>(d_ones, d_incoming_per_batch[layer][i], d_gradB[layer], cweight_rows, cweight_cols);
                    CU_CHECK(cudaMemcpy(d_totalBgrad + i * cweight_flat_size, d_gradB[layer], sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));

                    float *d_C_T; CU_CHECK(cudaMalloc(&d_C_T, cweight_flat_size * sizeof(float)));
                    transpose<<<calculate_grid_2d(cweight_cols, cweight_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_cweights[layer], d_C_T, cweight_rows, cweight_cols);
                    vecxmat2vec<<<gridOutGrad, block_1d>>>(d_incoming_per_batch[layer][i], d_C_T, d_preoutgoing_l, cweight_cols, cweight_rows);
                    dPower<<<gridOutGrad, block_1d>>>(d_activate_per_batch[layer - 1][i], d_dpow_l, order, cweight_rows);
                    sigmoidDer<<<gridOutGrad, block_1d>>>(d_dotProds_per_batch[layer - 1][i], d_dact_l, cweight_rows);
                    hadamard2<<<gridOutGrad, block_1d>>>(d_preoutgoing_l, d_dpow_l, d_dact_l, d_incoming_per_batch[layer - 1][i], 1, cweight_rows);
                    cudaFree(d_C_T);
                }

                matrix_vector_average<<<calculate_grid_2d(cweight_cols, cweight_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_totalCgrad, d_gradC[layer], batchSize, cweight_rows, cweight_cols);
                scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradC[layer], d_gradC[layer], ALPHA, cweight_flat_size);
                matrix_vector_average<<<calculate_grid_2d(cweight_cols, cweight_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_totalBgrad, d_gradB[layer], batchSize, cweight_rows, cweight_cols);
                scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradB[layer], d_gradB[layer], 1.0f - ALPHA, cweight_flat_size);
            }

            // First layer
            int cweight_rows_first = single_input_size, cweight_cols_first = width[0];
            size_t cweight_flat_size_first = cweight_rows_first * cweight_cols_first;
            dim3 gridWeightGradFirst = calculate_grid_1d(cweight_flat_size_first, WORKSIZE_1D);
            for (int i = 0; i < batchSize; ++i) {
                float* d_in_single; CU_CHECK(cudaMalloc(&d_in_single, single_input_size * sizeof(float)));
                CU_CHECK(cudaMemcpy(d_in_single, inputs[i].data(), single_input_size * sizeof(float), cudaMemcpyHostToDevice));
                vecxvec2mat<<<gridWeightGradFirst, block_1d>>>(d_in_single, d_incoming_per_batch[0][i], d_gradC[0], cweight_rows_first, cweight_cols_first);
                CU_CHECK(cudaMemcpy(d_totalCgrad + i * cweight_flat_size_first, d_gradC[0], sizeof(float) * cweight_flat_size_first, cudaMemcpyDeviceToDevice));
                vecxvec2mat<<<gridWeightGradFirst, block_1d>>>(d_ones, d_incoming_per_batch[0][i], d_gradB[0], cweight_rows_first, cweight_cols_first);
                CU_CHECK(cudaMemcpy(d_totalBgrad + i * cweight_flat_size_first, d_gradB[0], sizeof(float) * cweight_flat_size_first, cudaMemcpyDeviceToDevice));
                cudaFree(d_in_single);
            }

            matrix_vector_average<<<calculate_grid_2d(cweight_cols_first, cweight_rows_first, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_totalCgrad, d_gradC[0], batchSize, cweight_rows_first, cweight_cols_first);
            scaleByValue<<<gridWeightGradFirst, block_1d>>>(d_gradC[0], d_gradC[0], ALPHA, cweight_flat_size_first);
            matrix_vector_average<<<calculate_grid_2d(cweight_cols_first, cweight_rows_first, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_totalBgrad, d_gradB[0], batchSize, cweight_rows_first, cweight_cols_first);
            scaleByValue<<<gridWeightGradFirst, block_1d>>>(d_gradB[0], d_gradB[0], 1.0f - ALPHA, cweight_flat_size_first);

            // Update weights
            for (int i = 0; i < layers; ++i) {
                size_t c_size = cweights[i].size() * cweights[i][0].size();
                size_t b_size = bweights[i].size() * bweights[i][0].size();
                kernelUpdateWeightsElasticNet<<<calculate_grid_1d(c_size, WORKSIZE_1D), block_1d>>>(d_cweights[i], d_gradC[i], c_size, learningRate, LAMBDA_L1, LAMBDA_L2);
                kernelUpdateWeightsElasticNet<<<calculate_grid_1d(b_size, WORKSIZE_1D), block_1d>>>(d_bweights[i], d_gradB[i], b_size, learningRate, LAMBDA_L1, LAMBDA_L2);
            }
            CU_CHECK(cudaDeviceSynchronize());

            // Read back updated weights
            for (int i = 0; i < layers; ++i) {
                std::vector<float> c_upd(cweights[i].size() * cweights[i][0].size());
                std::vector<float> b_upd(bweights[i].size() * bweights[i][0].size());
                CU_CHECK(cudaMemcpy(c_upd.data(), d_cweights[i], c_upd.size() * sizeof(float), cudaMemcpyDeviceToHost));
                CU_CHECK(cudaMemcpy(b_upd.data(), d_bweights[i], b_upd.size() * sizeof(float), cudaMemcpyDeviceToHost));
                cweights[i] = reshape(c_upd, cweights[i].size(), cweights[i][0].size());
                bweights[i] = reshape(b_upd, bweights[i].size(), bweights[i][0].size());
            }

            for(int i=0; i<batchSize; ++i) cudaFree(d_err_per_batch[i]);
            for(int i=0; i<layers; ++i) for(int j=0; j<batchSize; ++j) {
                cudaFree(d_incoming_per_batch[i][j]);
                cudaFree(d_dotProds_per_batch[i][j]);
                cudaFree(d_activate_per_batch[i][j]);
            }
        }
        this->learningRate = initialLR;

    } catch (const std::runtime_error& e) {
        std::cerr << "Error during cuBufTrainBatch (mnn): " << e.what() << std::endl;
    }

    // --- Buffer Cleanup ---
    cudaFree(d_input_batch); cudaFree(d_target_batch);
    cudaFree(d_totalCgrad); cudaFree(d_totalBgrad);
    cudaFree(d_ones); cudaFree(d_preoutgoing_l); cudaFree(d_dpow_l); cudaFree(d_dact_l);
    for (int i = 0; i < layers; ++i) {
        cudaFree(d_cweights[i]); cudaFree(d_bweights[i]);
        cudaFree(d_gradC[i]); cudaFree(d_gradB[i]);
        cudaFree(d_dotProds[i]); cudaFree(d_activate[i]);
    }
}

/**
 * @brief Trains the mnn2d network on a batch of data with optimized buffer management.
 * @param inputs A vector of input matrices.
 * @param targets A vector of target vectors.
 */
void mnn2d::cuBufTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets) {
    if (inputs.size() != targets.size() || inputs.empty()) {
        throw std::invalid_argument("Invalid batch training data.");
    }
    this->batchSize = inputs.size();

    // --- Buffer Allocation ---
    float *d_input_batch = nullptr, *d_target_batch = nullptr, *d_final_output = nullptr;
    std::vector<float*> d_cweights(layers), d_bweights(layers), d_gradC(layers), d_gradB(layers);
    std::vector<float*> d_dotProds(layers), d_activate(layers);
    float *d_totalCgrad = nullptr, *d_totalBgrad = nullptr;
    float *d_grad_x_CT = nullptr, *d_dprev_p = nullptr, *d_dprev_act = nullptr;

    try {
        dim3 block_1d(WORKSIZE_1D);
        dim3 block_2d(WORKSIZE_2D_X, WORKSIZE_2D_Y);

        std::vector<float> flat_inputs;
        for(const auto& mat : inputs) for(const auto& row : mat) flat_inputs.insert(flat_inputs.end(), row.begin(), row.end());
        std::vector<float> flat_targets;
        for(const auto& vec : targets) flat_targets.insert(flat_targets.end(), vec.begin(), vec.end());

        CU_CHECK(cudaMalloc(&d_input_batch, flat_inputs.size() * sizeof(float)));
        CU_CHECK(cudaMemcpy(d_input_batch, flat_inputs.data(), flat_inputs.size() * sizeof(float), cudaMemcpyHostToDevice));
        CU_CHECK(cudaMalloc(&d_target_batch, flat_targets.size() * sizeof(float)));
        CU_CHECK(cudaMemcpy(d_target_batch, flat_targets.data(), flat_targets.size() * sizeof(float), cudaMemcpyHostToDevice));
        CU_CHECK(cudaMalloc(&d_final_output, batchSize * outWidth * sizeof(float)));

        size_t max_total_grad_size = 0;
        size_t max_act_size = 0;
        for (int i = 0; i < layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            size_t layer_output_size = batchSize * inHeight * width[i];
            max_total_grad_size = std::max(max_total_grad_size, c_size);
            max_act_size = std::max(max_act_size, layer_output_size);

            CU_CHECK(cudaMalloc(&d_cweights[i], c_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_bweights[i], b_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_gradC[i], c_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_gradB[i], b_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_dotProds[i], layer_output_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_activate[i], layer_output_size * sizeof(float)));
        }
        CU_CHECK(cudaMalloc(&d_totalCgrad, max_total_grad_size * batchSize * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_totalBgrad, max_total_grad_size * batchSize * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_grad_x_CT, max_act_size * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_dprev_p, max_act_size * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_dprev_act, max_act_size * sizeof(float)));

        int totalEpochs = 0;
        if (this->epochs < 1) this->epochs = 100;
        float initialLR = this->learningRate;

        // --- Training Loop ---
        while (true) {
            totalEpochs++;

            for (int i = 0; i < layers; ++i) {
                std::vector<float> flat_c = flatten(cweights[i]);
                std::vector<float> flat_b = flatten(bweights[i]);
                CU_CHECK(cudaMemcpy(d_cweights[i], flat_c.data(), flat_c.size() * sizeof(float), cudaMemcpyHostToDevice));
                CU_CHECK(cudaMemcpy(d_bweights[i], flat_b.data(), flat_b.size() * sizeof(float), cudaMemcpyHostToDevice));
            }

            // --- Forward Propagation ---
            float* d_current_act = d_input_batch;
            int single_input_height = inputs[0].size();
            int single_input_width = inputs[0][0].size();

            kernelLayerForwardBatch4<<<dim3(batchSize, single_input_height, width[0]), dim3(1, 1, 1)>>>(
                d_current_act, d_dotProds[0], d_cweights[0], d_bweights[0],
                batchSize, single_input_height, single_input_width, width[0], order
            );
            size_t dotprod_size_layer0 = batchSize * inHeight * width[0];
            softmax<<<calculate_grid_1d(dotprod_size_layer0, WORKSIZE_1D), block_1d>>>(d_dotProds[0], d_activate[0], SOFTMAX_TEMP, dotprod_size_layer0);

            for (int i = 1; i < layers; ++i) {
                d_current_act = d_activate[i - 1];
                kernelLayerForwardBatch4<<<dim3(batchSize, inHeight, width[i]), dim3(1, 1, 1)>>>(
                    d_current_act, d_dotProds[i], d_cweights[i], d_bweights[i],
                    batchSize, inHeight, width[i-1], width[i], order
                );
                size_t dotprod_size_layer_i = batchSize * inHeight * width[i];
                softmax<<<calculate_grid_1d(dotprod_size_layer_i, WORKSIZE_1D), block_1d>>>(d_dotProds[i], d_activate[i], SOFTMAX_TEMP, dotprod_size_layer_i);
            }

            meanPool<<<dim3(batchSize, outWidth), dim3(1, 1)>>>(d_activate[layers - 1], d_final_output, inHeight, outWidth, batchSize);
            CU_CHECK(cudaDeviceSynchronize());

            std::vector<float> final_output_flat(batchSize * outWidth);
            CU_CHECK(cudaMemcpy(final_output_flat.data(), d_final_output, sizeof(float) * final_output_flat.size(), cudaMemcpyDeviceToHost));
            for (int i = 0; i < batchSize; ++i) {
                std::copy(final_output_flat.begin() + i * outWidth, final_output_flat.begin() + (i + 1) * outWidth, outputBatch[i].begin());
            }

            for(int i=0; i<layers; ++i) {
                std::vector<float> flat_dots(batchSize * inHeight * width[i]), flat_acts(batchSize * inHeight * width[i]);
                CU_CHECK(cudaMemcpy(flat_dots.data(), d_dotProds[i], sizeof(float) * flat_dots.size(), cudaMemcpyDeviceToHost));
                CU_CHECK(cudaMemcpy(flat_acts.data(), d_activate[i], sizeof(float) * flat_acts.size(), cudaMemcpyDeviceToHost));
                for(int j=0; j<batchSize; ++j) {
                    dotBatch[i][j] = reshape(std::vector<float>(flat_dots.begin() + j*inHeight*width[i], flat_dots.begin() + (j+1)*inHeight*width[i]), inHeight, width[i]);
                    actBatch[i][j] = reshape(std::vector<float>(flat_acts.begin() + j*inHeight*width[i], flat_acts.begin() + (j+1)*inHeight*width[i]), inHeight, width[i]);
                }
            }

            int correct_predictions = 0;
            float total_loss = 0.0f;
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (maxIndex(outputBatch[i]) == maxIndex(targets[i])) correct_predictions++;
                total_loss += crossEntropy(outputBatch[i], targets[i]);
            }
            currloss = total_loss / batchSize;

            if (correct_predictions == inputs.size()) {
                std::cout << "All " << inputs.size() << " outputs correct after " << totalEpochs << " epochs. Loss: " << currloss << std::endl;
                break;
            } else {
                std::cout << " | Epoch: " << totalEpochs << "\tPredictions: " << correct_predictions << "/" << inputs.size()
                          << "\tAvg. CE Loss: " << currloss << std::endl;
            }

            if (totalEpochs >= this->epochs) {
                std::cout << "Epoch limit reached. Continuing training." << std::endl;
                this->epochs += 10;
            }

            // --- Backward Propagation ---
            std::vector<float*> d_err_per_batch(batchSize);
            std::vector<std::vector<float*>> d_incoming_per_batch(layers), d_dotProds_per_batch(layers), d_activate_per_batch(layers);
            for(int i=0; i<layers; ++i) {
                d_incoming_per_batch[i].resize(batchSize);
                d_dotProds_per_batch[i].resize(batchSize);
                d_activate_per_batch[i].resize(batchSize);
                for(int j=0; j<batchSize; ++j) {
                    size_t act_size = inHeight * width[i];
                    CU_CHECK(cudaMalloc(&d_incoming_per_batch[i][j], act_size * sizeof(float)));
                    CU_CHECK(cudaMalloc(&d_dotProds_per_batch[i][j], act_size * sizeof(float)));
                    CU_CHECK(cudaMalloc(&d_activate_per_batch[i][j], act_size * sizeof(float)));
                    CU_CHECK(cudaMemcpy(d_dotProds_per_batch[i][j], flatten(dotBatch[i][j]).data(), act_size * sizeof(float), cudaMemcpyHostToDevice));
                    CU_CHECK(cudaMemcpy(d_activate_per_batch[i][j], flatten(actBatch[i][j]).data(), act_size * sizeof(float), cudaMemcpyHostToDevice));
                }
            }

            for (int i = 0; i < batchSize; ++i) {
                CU_CHECK(cudaMalloc(&d_err_per_batch[i], targets[i].size() * sizeof(float)));
                float *d_out_single, *d_exp_single;
                CU_CHECK(cudaMalloc(&d_out_single, outputBatch[i].size() * sizeof(float)));
                CU_CHECK(cudaMalloc(&d_exp_single, targets[i].size() * sizeof(float)));
                CU_CHECK(cudaMemcpy(d_out_single, outputBatch[i].data(), outputBatch[i].size() * sizeof(float), cudaMemcpyHostToDevice));
                CU_CHECK(cudaMemcpy(d_exp_single, targets[i].data(), targets[i].size() * sizeof(float), cudaMemcpyHostToDevice));
                subtract<<<calculate_grid_1d(targets[i].size(), WORKSIZE_1D), block_1d>>>(d_out_single, d_exp_single, d_err_per_batch[i], targets[i].size());
                
                size_t last_layer_rows = actBatch[layers-1][i].size();
                size_t last_layer_cols = actBatch[layers-1][i][0].size();
                for(size_t r = 0; r < last_layer_rows; ++r) {
                    CU_CHECK(cudaMemcpy(d_incoming_per_batch[layers-1][i] + r * last_layer_cols, d_err_per_batch[i], sizeof(float) * last_layer_cols, cudaMemcpyDeviceToDevice));
                }
                cudaFree(d_out_single); cudaFree(d_exp_single);
            }

            for (int layer = layers - 1; layer >= 1; --layer) {
                int cweight_rows = cweights[layer].size(), cweight_cols = cweights[layer][0].size();
                size_t cweight_flat_size = cweight_rows * cweight_cols;

                for (int i = 0; i < batchSize; ++i) {
                    int prev_rows = actBatch[layer-1][i].size(), prev_cols = actBatch[layer-1][i][0].size();
                    int curr_rows = actBatch[layer][i].size(), curr_cols = actBatch[layer][i][0].size();
                    size_t prev_act_flat_size = prev_rows * prev_cols;

                    float *d_C_T; CU_CHECK(cudaMalloc(&d_C_T, cweight_flat_size * sizeof(float)));
                    transpose<<<calculate_grid_2d(cweight_cols, cweight_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_cweights[layer], d_C_T, cweight_rows, cweight_cols);
                    matxmat2mat<<<calculate_grid_2d(prev_cols, curr_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_incoming_per_batch[layer][i], d_C_T, d_grad_x_CT, curr_rows, curr_cols, prev_cols);
                    dPower<<<calculate_grid_1d(prev_act_flat_size, WORKSIZE_1D), block_1d>>>(d_activate_per_batch[layer-1][i], d_dprev_p, order, prev_act_flat_size);
                    softmaxDer<<<calculate_grid_1d(prev_act_flat_size, WORKSIZE_1D), block_1d>>>(d_dotProds_per_batch[layer-1][i], d_dprev_act, SOFTMAX_TEMP, prev_act_flat_size);
                    hadamard2<<<calculate_grid_1d(prev_act_flat_size, WORKSIZE_1D), block_1d>>>(d_grad_x_CT, d_dprev_p, d_dprev_act, d_incoming_per_batch[layer-1][i], prev_rows, prev_cols);

                    float *d_prev_p, *d_prev_p_T;
                    CU_CHECK(cudaMalloc(&d_prev_p, prev_act_flat_size * sizeof(float)));
                    CU_CHECK(cudaMalloc(&d_prev_p_T, prev_act_flat_size * sizeof(float)));
                    power<<<calculate_grid_1d(prev_act_flat_size, WORKSIZE_1D), block_1d>>>(d_activate_per_batch[layer-1][i], d_prev_p, order, prev_act_flat_size);
                    transpose<<<calculate_grid_2d(prev_cols, prev_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_prev_p, d_prev_p_T, prev_rows, prev_cols);
                    matxmat2mat<<<calculate_grid_2d(curr_cols, prev_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_prev_p_T, d_incoming_per_batch[layer][i], d_gradC[layer], prev_cols, prev_rows, curr_cols);
                    CU_CHECK(cudaMemcpy(d_totalCgrad + i * cweight_flat_size, d_gradC[layer], sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));

                    float* d_onesT; CU_CHECK(cudaMalloc(&d_onesT, prev_act_flat_size * sizeof(float)));
                    std::vector<float> h_ones(prev_act_flat_size, 1.0f);
                    CU_CHECK(cudaMemcpy(d_onesT, h_ones.data(), prev_act_flat_size * sizeof(float), cudaMemcpyHostToDevice));
                    matxmat2mat<<<calculate_grid_2d(curr_cols, prev_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_onesT, d_incoming_per_batch[layer][i], d_gradB[layer], prev_cols, prev_rows, curr_cols);
                    CU_CHECK(cudaMemcpy(d_totalBgrad + i * cweight_flat_size, d_gradB[layer], sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));
                    cudaFree(d_C_T); cudaFree(d_prev_p); cudaFree(d_prev_p_T); cudaFree(d_onesT);
                }
                matrix_vector_average<<<calculate_grid_2d(cweight_cols, cweight_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_totalCgrad, d_gradC[layer], batchSize, cweight_rows, cweight_cols);
                scaleByValue<<<calculate_grid_1d(cweight_flat_size, WORKSIZE_1D), block_1d>>>(d_gradC[layer], d_gradC[layer], ALPHA, cweight_flat_size);
                matrix_vector_average<<<calculate_grid_2d(cweight_cols, cweight_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_totalBgrad, d_gradB[layer], batchSize, cweight_rows, cweight_cols);
                scaleByValue<<<calculate_grid_1d(cweight_flat_size, WORKSIZE_1D), block_1d>>>(d_gradB[layer], d_gradB[layer], 1.0f - ALPHA, cweight_flat_size);
            }

            // First layer
            int cweight_rows_first = inWidth, cweight_cols_first = width[0];
            size_t cweight_flat_size_first = cweight_rows_first * cweight_cols_first;
            for (int i = 0; i < batchSize; ++i) {
                float* d_input_single; CU_CHECK(cudaMalloc(&d_input_single, inHeight * inWidth * sizeof(float)));
                CU_CHECK(cudaMemcpy(d_input_single, flatten(inputs[i]).data(), inHeight * inWidth * sizeof(float), cudaMemcpyHostToDevice));
                float *d_input_p, *d_input_p_T;
                CU_CHECK(cudaMalloc(&d_input_p, inHeight * inWidth * sizeof(float)));
                CU_CHECK(cudaMalloc(&d_input_p_T, inHeight * inWidth * sizeof(float)));
                power<<<calculate_grid_1d(inHeight * inWidth, WORKSIZE_1D), block_1d>>>(d_input_single, d_input_p, order, inHeight * inWidth);
                transpose<<<calculate_grid_2d(inWidth, inHeight, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_input_p, d_input_p_T, inHeight, inWidth);
                matxmat2mat<<<calculate_grid_2d(cweight_cols_first, inHeight, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_input_p_T, d_incoming_per_batch[0][i], d_gradC[0], inWidth, inHeight, cweight_cols_first);
                CU_CHECK(cudaMemcpy(d_totalCgrad + i * cweight_flat_size_first, d_gradC[0], sizeof(float) * cweight_flat_size_first, cudaMemcpyDeviceToDevice));

                float* d_onesT; CU_CHECK(cudaMalloc(&d_onesT, inHeight * inWidth * sizeof(float)));
                std::vector<float> h_ones(inHeight * inWidth, 1.0f);
                CU_CHECK(cudaMemcpy(d_onesT, h_ones.data(), inHeight * inWidth * sizeof(float), cudaMemcpyHostToDevice));
                matxmat2mat<<<calculate_grid_2d(cweight_cols_first, inHeight, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_onesT, d_incoming_per_batch[0][i], d_gradB[0], inWidth, inHeight, cweight_cols_first);
                CU_CHECK(cudaMemcpy(d_totalBgrad + i * cweight_flat_size_first, d_gradB[0], sizeof(float) * cweight_flat_size_first, cudaMemcpyDeviceToDevice));
                cudaFree(d_input_single); cudaFree(d_input_p); cudaFree(d_input_p_T); cudaFree(d_onesT);
            }
            matrix_vector_average<<<calculate_grid_2d(cweight_cols_first, cweight_rows_first, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_totalCgrad, d_gradC[0], batchSize, cweight_rows_first, cweight_cols_first);
            scaleByValue<<<calculate_grid_1d(cweight_flat_size_first, WORKSIZE_1D), block_1d>>>(d_gradC[0], d_gradC[0], ALPHA, cweight_flat_size_first);
            matrix_vector_average<<<calculate_grid_2d(cweight_cols_first, cweight_rows_first, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_totalBgrad, d_gradB[0], batchSize, cweight_rows_first, cweight_cols_first);
            scaleByValue<<<calculate_grid_1d(cweight_flat_size_first, WORKSIZE_1D), block_1d>>>(d_gradB[0], d_gradB[0], 1.0f - ALPHA, cweight_flat_size_first);

            // Update weights
            for (int i = 0; i < layers; ++i) {
                size_t c_size = cweights[i].size() * cweights[i][0].size();
                size_t b_size = bweights[i].size() * bweights[i][0].size();
                kernelUpdateWeightsElasticNet<<<calculate_grid_1d(c_size, WORKSIZE_1D), block_1d>>>(d_cweights[i], d_gradC[i], c_size, learningRate, LAMBDA_L1, LAMBDA_L2);
                kernelUpdateWeightsElasticNet<<<calculate_grid_1d(b_size, WORKSIZE_1D), block_1d>>>(d_bweights[i], d_gradB[i], b_size, learningRate, LAMBDA_L1, LAMBDA_L2);
            }
            CU_CHECK(cudaDeviceSynchronize());

            for(int i=0; i<batchSize; ++i) cudaFree(d_err_per_batch[i]);
            for(int i=0; i<layers; ++i) for(int j=0; j<batchSize; ++j) {
                cudaFree(d_incoming_per_batch[i][j]);
                cudaFree(d_dotProds_per_batch[i][j]);
                cudaFree(d_activate_per_batch[i][j]);
            }
        }
        this->learningRate = initialLR;

    } catch (const std::runtime_error& e) {
        std::cerr << "Error during cuBufTrainBatch (mnn2d): " << e.what() << std::endl;
    }

    // --- Buffer Cleanup ---
    cudaFree(d_input_batch); cudaFree(d_target_batch); cudaFree(d_final_output);
    cudaFree(d_totalCgrad); cudaFree(d_totalBgrad);
    cudaFree(d_grad_x_CT); cudaFree(d_dprev_p); cudaFree(d_dprev_act);
    for (int i = 0; i < layers; ++i) {
        cudaFree(d_cweights[i]); cudaFree(d_bweights[i]);
        cudaFree(d_gradC[i]); cudaFree(d_gradB[i]);
        cudaFree(d_dotProds[i]); cudaFree(d_activate[i]);
    }
}

#endif // USE_CU