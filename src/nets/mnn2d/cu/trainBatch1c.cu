#ifdef USE_CU
#include "mnn1d.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cstdlib> // For rand()
#include <algorithm> // For std::max, std::copy

void mnn2d::cuTrainBatch1c(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>> &targets, bool useThreadOrBuffer) {
    if (inputs.size() != targets.size() || inputs.empty()) {
        throw std::invalid_argument("Invalid batch training data.");
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
                int cols = width[i];
                dotBatch[i][j].resize(inHeight, std::vector<float>(cols));
                actBatch[i][j].resize(inHeight, std::vector<float>(cols));
            }
        }
    }
    if (outputBatch.size() != batchSize) {
        outputBatch.resize(batchSize);
        for(int i=0; i<batchSize; ++i) outputBatch[i].resize(outSize);
    }

    if (!useThreadOrBuffer) {
        cuForprop(inputs);
        int correct_predictions = 0;
        float total_loss = 0.0f;
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (maxIndex(outputBatch[i]) == maxIndex(targets[i])) {
                correct_predictions++;
            }
            total_loss += crossEntropy(outputBatch[i], targets[i]);
        }
        currloss = total_loss / batchSize;

        if (correct_predictions == inputs.size()) {
            std::cout << "All " << inputs.size() << " outputs correct. Loss: " << currloss << std::endl;
        } else {
            std::cout << " | Predictions: " << correct_predictions << "/" << inputs.size()
                      << "\tAvg. CE Loss: " << currloss << std::endl;
            cuBackprop(targets);
        }
    }
    else {
        // --- Buffer Allocation ---
        float *d_input_batch = nullptr, *d_target_batch = nullptr, *d_final_output = nullptr;
        std::vector<float*> d_cweights(layers), d_bweights(layers), d_gradC(layers), d_gradB(layers);
        std::vector<float*> d_dotProds(layers), d_activate(layers);
        float *d_totalCgrad = nullptr, *d_totalBgrad = nullptr;
        float *d_grad_x_CT = nullptr, *d_dprev_p = nullptr, *d_dprev_act = nullptr;
        dim3 block_1d(WORKSIZE_1D);
        dim3 block_2d(WORKSIZE_2D_X, WORKSIZE_2D_Y);

        try {
            std::vector<float> flat_inputs;
            for(const auto& mat : inputs) for(const auto& row : mat) flat_inputs.insert(flat_inputs.end(), row.begin(), row.end());
            std::vector<float> flat_targets;
            for(const auto& vec : targets) flat_targets.insert(flat_targets.end(), vec.begin(), vec.end());

            CU_CHECK(cudaMalloc(&d_input_batch, flat_inputs.size() * sizeof(float)));
            CU_CHECK(cudaMemcpy(d_input_batch, flat_inputs.data(), flat_inputs.size() * sizeof(float), cudaMemcpyHostToDevice));
            CU_CHECK(cudaMalloc(&d_target_batch, flat_targets.size() * sizeof(float)));
            CU_CHECK(cudaMemcpy(d_target_batch, flat_targets.data(), flat_targets.size() * sizeof(float), cudaMemcpyHostToDevice));
            CU_CHECK(cudaMalloc(&d_final_output, batchSize * outSize * sizeof(float)));

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

            kernelLayerForwardBatch4<<<dim3(batchSize, single_input_height, width[0]), dim3(1, 1, 1)>>>(d_current_act, d_dotProds[0], d_cweights[0], d_bweights[0], batchSize, single_input_height, single_input_width, width[0], order);
            size_t dotprod_size_layer0 = batchSize * inHeight * width[0];
            relu<<<calculate_grid_1d(dotprod_size_layer0, WORKSIZE_1D), block_1d>>>(d_dotProds[0], d_activate[0], dotprod_size_layer0);

            for (int i = 1; i < layers; ++i) {
                d_current_act = d_activate[i - 1];
                kernelLayerForwardBatch4<<<dim3(batchSize, inHeight, width[i]), dim3(1, 1, 1)>>>(d_current_act, d_dotProds[i], d_cweights[i], d_bweights[i], batchSize, inHeight, width[i-1], width[i], order);
                size_t dotprod_size_layer_i = batchSize * inHeight * width[i];
                relu<<<calculate_grid_1d(dotprod_size_layer_i, WORKSIZE_1D), block_1d>>>(d_dotProds[i], d_activate[i], dotprod_size_layer_i);
            }

            meanPool<<<dim3(batchSize, outSize), dim3(1, 1)>>>(d_activate[layers - 1], d_final_output, inHeight, outSize, batchSize);
            CU_CHECK(cudaDeviceSynchronize());

            std::vector<float> final_output_flat(batchSize * outSize);
            CU_CHECK(cudaMemcpy(final_output_flat.data(), d_final_output, sizeof(float) * final_output_flat.size(), cudaMemcpyDeviceToHost));
            for (int i = 0; i < batchSize; ++i) {
                std::copy(final_output_flat.begin() + i * outSize, final_output_flat.begin() + (i + 1) * outSize, outputBatch[i].begin());
                outputBatch[i] = softmax(outputBatch[i]);
            }

            int correct_predictions = 0;
            float total_loss = 0.0f;
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (maxIndex(outputBatch[i]) == maxIndex(targets[i])) correct_predictions++;
                total_loss += crossEntropy(outputBatch[i], targets[i]);
            }
            currloss = total_loss / batchSize;

            if (correct_predictions == inputs.size()) {
                std::cout << "All " << inputs.size() << " outputs correct. Loss: " << currloss << std::endl;
            }
            else {
                std::cout << " | Predictions: " << correct_predictions << "/" << inputs.size() << "\tAvg. CE Loss: " << currloss << std::endl;

                // --- Backward Propagation ---
                for(int i=0; i<layers; ++i) {
                    std::vector<float> flat_dots(batchSize * inHeight * width[i]), flat_acts(batchSize * inHeight * width[i]);
                    CU_CHECK(cudaMemcpy(flat_dots.data(), d_dotProds[i], sizeof(float) * flat_dots.size(), cudaMemcpyDeviceToHost));
                    CU_CHECK(cudaMemcpy(flat_acts.data(), d_activate[i], sizeof(float) * flat_acts.size(), cudaMemcpyDeviceToHost));
                    for(int j=0; j<batchSize; ++j) {
                        dotBatch[i][j] = reshape(std::vector<float>(flat_dots.begin() + j*inHeight*width[i], flat_dots.begin() + (j+1)*inHeight*width[i]), inHeight, width[i]);
                        actBatch[i][j] = reshape(std::vector<float>(flat_acts.begin() + j*inHeight*width[i], flat_acts.begin() + (j+1)*inHeight*width[i]), inHeight, width[i]);
                    }
                }

                std::vector<float*> d_err_per_batch(batchSize);
                std::vector<std::vector<float*>> d_incoming_per_batch(layers), d_dotProds_per_batch(layers), d_activate_per_batch(layers);
                for(int i=0; i<layers; ++i) {
                    d_incoming_per_batch[i].resize(batchSize); d_dotProds_per_batch[i].resize(batchSize); d_activate_per_batch[i].resize(batchSize);
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
                    float* src = d_final_output + i * outSize;
                    CU_CHECK(cudaMemcpy(d_out_single, src, targets[i].size() * sizeof(float), cudaMemcpyHostToDevice));
                    CU_CHECK(cudaMalloc(&d_exp_single, targets[i].size() * sizeof(float)));
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
                        // transpose
                        float *d_C_T; CU_CHECK(cudaMalloc(&d_C_T, cweight_flat_size * sizeof(float)));
                        transpose<<<calculate_grid_2d(cweight_cols, cweight_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_cweights[layer], d_C_T, cweight_rows, cweight_cols);
                        // dL/dz_l x C^T
                        matxmat2mat<<<calculate_grid_2d(prev_cols, curr_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_incoming_per_batch[layer][i], d_C_T, d_grad_x_CT, curr_rows, curr_cols, prev_cols);
                        // d(prev_p)
                        dPower<<<calculate_grid_1d(prev_act_flat_size, WORKSIZE_1D), block_1d>>>(d_activate_per_batch[layer-1][i], d_dprev_p, order, prev_act_flat_size);
                        // Calculate d(prev_act)
                        reluDer<<<calculate_grid_1d(prev_act_flat_size, WORKSIZE_1D), block_1d>>>(d_dotProds_per_batch[layer-1][i], d_dprev_act, prev_act_flat_size);
                        // outgoing = (dL/dz_l * C^T) . d(prev_p) . d(prev_act)
                        hadamard2<<<calculate_grid_1d(prev_act_flat_size, WORKSIZE_1D), block_1d>>>(d_grad_x_CT, d_dprev_p, d_dprev_act, d_incoming_per_batch[layer-1][i], prev_rows, prev_cols);

                        // --- Calculate Weight Gradients ---
                        // gradc = ALPHA * prev_p^T * incoming
                        // power prev_p
                        float *d_prev_p, *d_prev_p_T;
                        CU_CHECK(cudaMalloc(&d_prev_p, prev_act_flat_size * sizeof(float)));
                        CU_CHECK(cudaMalloc(&d_prev_p_T, prev_act_flat_size * sizeof(float)));
                        power<<<calculate_grid_1d(prev_act_flat_size, WORKSIZE_1D), block_1d>>>(d_activate_per_batch[layer-1][i], d_prev_p, order, prev_act_flat_size);
                        // transpose prev_p
                        transpose<<<calculate_grid_2d(prev_cols, prev_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_prev_p, d_prev_p_T, prev_rows, prev_cols);
                        // dL/dC_layer
                        matxmat2mat<<<calculate_grid_2d(curr_cols, prev_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_prev_p_T, d_incoming_per_batch[layer][i], d_gradC[layer], prev_cols, prev_rows, curr_cols);
                        CU_CHECK(cudaMemcpy(d_totalCgrad + i * cweight_flat_size, d_gradC[layer], sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));

                        // gradB = dL/dz_l x V1^T
                        // transpose ones = onesT
                        float* d_onesT; CU_CHECK(cudaMalloc(&d_onesT, prev_act_flat_size * sizeof(float)));
                        std::vector<float> h_ones(prev_act_flat_size, 1.0f);
                        CU_CHECK(cudaMemcpy(d_onesT, h_ones.data(), prev_act_flat_size * sizeof(float), cudaMemcpyHostToDevice));
                        // dL/dB_layer
                        matxmat2mat<<<calculate_grid_2d(curr_cols, prev_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_onesT, d_incoming_per_batch[layer][i], d_gradB[layer], prev_cols, prev_rows, curr_cols);
                        CU_CHECK(cudaMemcpy(d_totalBgrad + i * cweight_flat_size, d_gradB[layer], sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));
                        cudaFree(d_C_T); cudaFree(d_prev_p); cudaFree(d_prev_p_T); cudaFree(d_onesT);
                    }
                    // Average Gradients
                    matrix_vector_average<<<calculate_grid_2d(cweight_cols, cweight_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_totalCgrad, d_gradC[layer], batchSize, cweight_rows, cweight_cols);
                    // scale dL/dC_layer by ALPHA
                    scaleByValue<<<calculate_grid_1d(cweight_flat_size, WORKSIZE_1D), block_1d>>>(d_gradC[layer], d_gradC[layer], ALPHA, cweight_flat_size);
                    // Average Gradients
                    matrix_vector_average<<<calculate_grid_2d(cweight_cols, cweight_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_totalBgrad, d_gradB[layer], batchSize, cweight_rows, cweight_cols);
                    // scale dL/dB_layer by 1- ALPHA
                    scaleByValue<<<calculate_grid_1d(cweight_flat_size, WORKSIZE_1D), block_1d>>>(d_gradB[layer], d_gradB[layer], 1.0f - ALPHA, cweight_flat_size);
                }

                int cweight_rows_first = inWidth, cweight_cols_first = width[0];
                size_t cweight_flat_size_first = cweight_rows_first * cweight_cols_first;
                for (int i = 0; i < batchSize; ++i) {
                    float* d_input_single; CU_CHECK(cudaMalloc(&d_input_single, inHeight * inWidth * sizeof(float)));
                    CU_CHECK(cudaMemcpy(d_input_single, flatten(inputs[i]).data(), inHeight * inWidth * sizeof(float), cudaMemcpyHostToDevice));
                    float *d_input_p, *d_input_p_T;
                    CU_CHECK(cudaMalloc(&d_input_p, inHeight * inWidth * sizeof(float)));
                    CU_CHECK(cudaMalloc(&d_input_p_T, inHeight * inWidth * sizeof(float)));
                    // power input
                    power<<<calculate_grid_1d(inHeight * inWidth, WORKSIZE_1D), block_1d>>>(d_input_single, d_input_p, order, inHeight * inWidth);
                    // transpose input
                    transpose<<<calculate_grid_2d(inWidth, inHeight, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_input_p, d_input_p_T, inHeight, inWidth);
                    // dL/dC_0
                    matxmat2mat<<<calculate_grid_2d(cweight_cols_first, inHeight, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_input_p_T, d_incoming_per_batch[0][i], d_gradC[0], inWidth, inHeight, cweight_cols_first);
                    CU_CHECK(cudaMemcpy(d_totalCgrad + i * cweight_flat_size_first, d_gradC[0], sizeof(float) * cweight_flat_size_first, cudaMemcpyDeviceToDevice));

                    // dL/dB_0
                    float* d_onesT; CU_CHECK(cudaMalloc(&d_onesT, inHeight * inWidth * sizeof(float)));
                    std::vector<float> h_ones(inHeight * inWidth, 1.0f);
                    CU_CHECK(cudaMemcpy(d_onesT, h_ones.data(), inHeight * inWidth * sizeof(float), cudaMemcpyHostToDevice));
                    matxmat2mat<<<calculate_grid_2d(cweight_cols_first, inHeight, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_onesT, d_incoming_per_batch[0][i], d_gradB[0], inWidth, inHeight, cweight_cols_first);
                    CU_CHECK(cudaMemcpy(d_totalBgrad + i * cweight_flat_size_first, d_gradB[0], sizeof(float) * cweight_flat_size_first, cudaMemcpyDeviceToDevice));
                    cudaFree(d_input_single); cudaFree(d_input_p); cudaFree(d_input_p_T); cudaFree(d_onesT);
                }
                // Average the gradients
                matrix_vector_average<<<calculate_grid_2d(cweight_cols_first, cweight_rows_first, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_totalCgrad, d_gradC[0], batchSize, cweight_rows_first, cweight_cols_first);
                // scale by ALPHA
                scaleByValue<<<calculate_grid_1d(cweight_flat_size_first, WORKSIZE_1D), block_1d>>>(d_gradC[0], d_gradC[0], ALPHA, cweight_flat_size_first);
                // Average the gradients
                matrix_vector_average<<<calculate_grid_2d(cweight_cols_first, cweight_rows_first, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(d_totalBgrad, d_gradB[0], batchSize, cweight_rows_first, cweight_cols_first);
                // scale by 1
                scaleByValue<<<calculate_grid_1d(cweight_flat_size_first, WORKSIZE_1D), block_1d>>>(d_gradB[0], d_gradB[0], 1.0f - ALPHA, cweight_flat_size_first);

                for (int i = 0; i < layers; ++i) {
                    size_t c_size = cweights[i].size() * cweights[i][0].size();
                    size_t b_size = bweights[i].size() * bweights[i][0].size();
                switch (weightUpdateType) {
                    case 0:
                        kernelUpdateWeights<<<calculate_grid_1d(c_size, WORKSIZE_1D), block_1d>>>(d_cweights[i], d_gradC[i], learningRate, c_size);
                        kernelUpdateWeights<<<calculate_grid_1d(b_size, WORKSIZE_1D), block_1d>>>(d_bweights[i], d_gradB[i], learningRate, b_size);
                        break;
                    case 1:
                        kernelUpdateWeightsWithL1<<<calculate_grid_1d(c_size, WORKSIZE_1D), block_1d>>>(d_cweights[i], d_gradC[i], c_size, learningRate, LAMBDA_L1);
                        kernelUpdateWeightsWithL1<<<calculate_grid_1d(b_size, WORKSIZE_1D), block_1d>>>(d_bweights[i], d_gradB[i], b_size, learningRate, LAMBDA_L1);
                        break;
                    case 2:
                        kernelUpdateWeightsWithL2<<<calculate_grid_1d(c_size, WORKSIZE_1D), block_1d>>>(d_cweights[i], d_gradC[i], c_size, learningRate, LAMBDA_L2);
                        kernelUpdateWeightsWithL2<<<calculate_grid_1d(b_size, WORKSIZE_1D), block_1d>>>(d_bweights[i], d_gradB[i], b_size, learningRate, LAMBDA_L2);
                        break;
                    case 3:
                        kernelUpdateWeightsElasticNet<<<calculate_grid_1d(c_size, WORKSIZE_1D), block_1d>>>(d_cweights[i], d_gradC[i], c_size, learningRate, LAMBDA_L1, LAMBDA_L2);
                        kernelUpdateWeightsElasticNet<<<calculate_grid_1d(b_size, WORKSIZE_1D), block_1d>>>(d_bweights[i], d_gradB[i], b_size, learningRate, LAMBDA_L1, LAMBDA_L2);
                        break;
                    case 4:
                        kernelUpdateWeightsWithWeightDecay<<<calculate_grid_1d(c_size, WORKSIZE_1D), block_1d>>>(d_cweights[i], d_gradC[i], c_size, learningRate, WEIGHT_DECAY);
                        kernelUpdateWeightsWithWeightDecay<<<calculate_grid_1d(b_size, WORKSIZE_1D), block_1d>>>(d_bweights[i], d_gradB[i], b_size, learningRate, WEIGHT_DECAY);
                        break;
                    case 5:
                        kernelUpdateWeightsDropout<<<calculate_grid_1d(c_size, WORKSIZE_1D), block_1d>>>(d_cweights[i], d_gradC[i], c_size, learningRate, DROPOUT_RATE, (uint)rand());
                        kernelUpdateWeightsDropout<<<calculate_grid_1d(b_size, WORKSIZE_1D), block_1d>>>(d_bweights[i], d_gradB[i], b_size, learningRate, DROPOUT_RATE, (uint)rand());
                        break;
                }
                }
                CU_CHECK(cudaDeviceSynchronize());

                for(int i=0; i<batchSize; ++i) cudaFree(d_err_per_batch[i]);
                for(int i=0; i<layers; ++i) for(int j=0; j<batchSize; ++j) {
                    cudaFree(d_incoming_per_batch[i][j]);
                    cudaFree(d_dotProds_per_batch[i][j]);
                    cudaFree(d_activate_per_batch[i][j]);
                }
            }
        } catch (const std::runtime_error& e) {
            std::cerr << "Error during mnn2d::cuTrainBatch1c (CUDA): " << e.what() << std::endl;
        }

        cudaFree(d_input_batch); cudaFree(d_target_batch); cudaFree(d_final_output);
        cudaFree(d_totalCgrad); cudaFree(d_totalBgrad);
        cudaFree(d_grad_x_CT); cudaFree(d_dprev_p); cudaFree(d_dprev_act);
        for (int i = 0; i < layers; ++i) {
            cudaFree(d_cweights[i]); cudaFree(d_bweights[i]);
            cudaFree(d_gradC[i]); cudaFree(d_gradB[i]);
            cudaFree(d_dotProds[i]); cudaFree(d_activate[i]);
        }
    }
}

#endif