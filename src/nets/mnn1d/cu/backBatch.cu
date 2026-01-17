#ifdef USE_CU
#include "mnn1d.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstdlib>

/**
 * @brief batch backpropgation using CUDA for mnn1d
 * @param expected expected result from forprop
 */
void mnn1d::cuBackprop(const std::vector<std::vector<float>>& expected) {
    try {
        int batchSize = expected.size();
        if (batchSize == 0) {
            throw std::runtime_error("Batch size should be greater than 0.");
        }

        dim3 block_1d(WORKSIZE_1D);
        dim3 block_2d(WORKSIZE_2D_X, WORKSIZE_2D_Y);
        size_t inputSize = inputBatch[0].size();
        size_t outputSize = outputBatch[0].size();
        int max_layer_width = 0;
        for(int w : width) max_layer_width = std::max(max_layer_width, w);
        max_layer_width = std::max(max_layer_width, (int)inputSize);

        // --- Device Memory Pointers (Per Batch Element) ---
        std::vector<float*> d_in(batchSize), d_exp(batchSize), d_out(batchSize), d_err(batchSize);
        std::vector<std::vector<float*>> d_incoming(this->layers), d_dotProds(this->layers), d_activate(this->layers);
        for(int i = 0; i < layers; ++i) {
            d_incoming[i].resize(batchSize);
            d_dotProds[i].resize(batchSize);
            d_activate[i].resize(batchSize);
        }

        // --- Device Memory Pointers (Shared) ---
        std::vector<float*> d_cweights(this->layers), d_bweights(this->layers);
        std::vector<float*> d_gradC(this->layers), d_gradB(this->layers);
        float *d_ones = nullptr;

        // --- Allocate and Copy Per-Batch Data ---
        for (int i = 0; i < batchSize; i++) {
            size_t batch_input_size = inputBatch[i].size();
            size_t batch_output_size = outputBatch[i].size();

            CU_CHECK(cudaMalloc(&d_in[i], sizeof(float) * batch_input_size));
            CU_CHECK(cudaMalloc(&d_exp[i], sizeof(float) * batch_output_size));
            CU_CHECK(cudaMalloc(&d_out[i], sizeof(float) * batch_output_size));
            CU_CHECK(cudaMalloc(&d_err[i], sizeof(float) * batch_output_size));

            CU_CHECK(cudaMemcpy(d_in[i], inputBatch[i].data(), sizeof(float) * batch_input_size, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(d_exp[i], expected[i].data(), sizeof(float) * batch_output_size, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(d_out[i], actBatch[layers-1][i].data(), sizeof(float) * batch_output_size, cudaMemcpyHostToDevice));
        }

        // Initial Error Calculation
        for (int i = 0; i < batchSize; i++) {
            size_t batch_output_size = actBatch[layers-1][i].size();
            dim3 gridSub = calculate_grid_1d(batch_output_size, WORKSIZE_1D);
            subtract<<<gridSub, block_1d>>>(d_out[i], d_exp[i], d_err[i], (int)batch_output_size);
        }

        // --- Allocate and Copy Shared Data & Batch Activations ---
        size_t max_total_grad_size = 0;
        for(int i = 0; i < this->layers; i++) {
            std::vector<float> flat_c = flatten(cweights[i]);
            std::vector<float> flat_b = flatten(bweights[i]);
            size_t c_size = flat_c.size();
            max_total_grad_size = std::max(max_total_grad_size, c_size);
            CU_CHECK(cudaMalloc(&d_cweights[i], sizeof(float) * c_size));
            CU_CHECK(cudaMalloc(&d_bweights[i], sizeof(float) * flat_b.size()));
            CU_CHECK(cudaMalloc(&d_gradC[i], sizeof(float) * c_size));
            CU_CHECK(cudaMalloc(&d_gradB[i], sizeof(float) * flat_b.size()));
            CU_CHECK(cudaMemcpy(d_cweights[i], flat_c.data(), sizeof(float) * c_size, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(d_bweights[i], flat_b.data(), sizeof(float) * flat_b.size(), cudaMemcpyHostToDevice));
            for(int j = 0; j < batchSize; j++) {
                size_t act_size = actBatch[i][j].size();
                CU_CHECK(cudaMalloc(&d_activate[i][j], sizeof(float) * act_size));
                CU_CHECK(cudaMalloc(&d_dotProds[i][j], sizeof(float) * act_size));
                CU_CHECK(cudaMalloc(&d_incoming[i][j], sizeof(float) * act_size));
                CU_CHECK(cudaMemcpy(d_activate[i][j], actBatch[i][j].data(), sizeof(float) * act_size, cudaMemcpyHostToDevice));
                CU_CHECK(cudaMemcpy(d_dotProds[i][j], dotBatch[i][j].data(), sizeof(float) * act_size, cudaMemcpyHostToDevice));
                if (i == layers - 1) { // Copy initial error to last layer's incoming gradient
                    CU_CHECK(cudaMemcpy(d_incoming[i][j], d_err[j], sizeof(float) * act_size, cudaMemcpyDeviceToDevice));
                }
            }
        }

        // Allocate buffers for average gradient accumulation (Total size = max_total_grad_size * batchSize)
        float *d_totalCgrad = nullptr, *d_totalBgrad = nullptr;
        CU_CHECK(cudaMalloc(&d_totalCgrad, max_total_grad_size * batchSize * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_totalBgrad, max_total_grad_size * batchSize * sizeof(float)));

        // Allocate shared intermediate buffers for outgoing gradient calculation
        float *d_preoutgoing_l = nullptr, *d_outgoing_l = nullptr;
        float *d_dpow_l = nullptr, *d_dact_l = nullptr;
        CU_CHECK(cudaMalloc(&d_preoutgoing_l, max_layer_width * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_outgoing_l, max_layer_width * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_dpow_l, max_layer_width * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_dact_l, max_layer_width * sizeof(float)));

        // Allocate and copy ones vector
        std::vector<float> v1(max_layer_width, 1.0f);
        CU_CHECK(cudaMalloc(&d_ones, sizeof(float) * max_layer_width));
        CU_CHECK(cudaMemcpy(d_ones, v1.data(), sizeof(float) * max_layer_width, cudaMemcpyHostToDevice));

        // --- Backpropagation Loop (Last Layer to Second Layer) ---
        for(int layer = layers - 1; layer >= 1; --layer) {
            int prev_size = actBatch[layer - 1][0].size();
            int curr_size = actBatch[layer][0].size();
            int cweight_rows = prev_size;
            int cweight_cols = curr_size;
            size_t cweight_flat_size = cweight_rows * cweight_cols;
            dim3 gridWeightGrad = calculate_grid_1d(cweight_flat_size, WORKSIZE_1D);
            dim3 gridOutGrad = calculate_grid_1d(prev_size, WORKSIZE_1D);

            for(int i = 0; i < batchSize; ++i) {
                // 1. dL/dC_l (vecxvec2mat: d_activate[L-1][i] x d_incoming[L][i])
                vecxvec2mat<<<gridWeightGrad, block_1d>>>(
                    d_activate[layer - 1][i], d_incoming[layer][i], d_gradC[layer], cweight_rows, cweight_cols
                );
                // Copy to total C grad buffer
                CU_CHECK(cudaMemcpy(d_totalCgrad + i * cweight_flat_size, d_gradC[layer], sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));

                // 2. dL/dB_l (vecxvec2mat: d_ones x d_incoming[L][i])
                vecxvec2mat<<<gridWeightGrad, block_1d>>>(
                    d_ones, d_incoming[layer][i], d_gradB[layer], cweight_rows, cweight_cols
                );
                // Copy to total B grad buffer
                CU_CHECK(cudaMemcpy(d_totalBgrad + i * cweight_flat_size, d_gradB[layer], sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));

                // 3. Outgoing Gradient Calculation
                float *d_C_T = nullptr;
                CU_CHECK(cudaMalloc(&d_C_T, sizeof(float) * cweight_flat_size));
                transpose<<<calculate_grid_2d(cweight_cols, cweight_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                    d_cweights[layer], d_C_T, cweight_rows, cweight_cols
                );
                // incoming gradient x C^T (vecxmat2vec)
                vecxmat2vec<<<gridOutGrad, block_1d>>>(
                    d_incoming[layer][i], d_C_T, d_preoutgoing_l, cweight_cols, cweight_rows
                );
                CU_CHECK(cudaFree(d_C_T));

                // derivative of power (dPower)
                dPower<<<gridOutGrad, block_1d>>>(d_activate[layer - 1][i], d_dpow_l, order, prev_size);

                // derivative of activation (sigmoidDer)
                sigmoidDer<<<gridOutGrad, block_1d>>>(d_dotProds[layer - 1][i], d_dact_l, prev_size);

                // outgoing gradient = d_preoutgoing . d_dpow . d_dact (hadamard2)
                hadamard2<<<gridOutGrad, block_1d>>>(
                    d_preoutgoing_l, d_dpow_l, d_dact_l, d_incoming[layer - 1][i], 1, prev_size
                );
            }

            // 4. Average the Gradients (Overwrite d_gradC[layer] with averaged value)
            int rows = cgradients[layer].size();
            int cols = cgradients[layer][0].size();
            matrix_vector_average<<<calculate_grid_2d(cols, rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), dim3(WORKSIZE_2D_X, WORKSIZE_2D_Y)>>>(
                d_totalCgrad, d_gradC[layer], batchSize, rows, cols
            );
            matrix_vector_average<<<calculate_grid_2d(cols, rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), dim3(WORKSIZE_2D_X, WORKSIZE_2D_Y)>>>(
                d_totalBgrad, d_gradB[layer], batchSize, rows, cols
            );
            
            // Scale by ALPHA / 1-ALPHA (done during averaging/update in CL, but here it's separate)
            scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradC[layer], d_gradC[layer], ALPHA, (int)cweight_flat_size);
            scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradB[layer], d_gradB[layer], 1.0f - ALPHA, (int)cweight_flat_size); // CL uses ALPHA for both
        }

        // --- Backpropagation for the First Layer (Layer 0) ---
        int cweight_rows = inputSize;
        int cweight_cols = cweights[0][0].size();
        size_t cweight_flat_size = cweight_rows * cweight_cols;
        dim3 gridWeightGradFirst = calculate_grid_1d(cweight_flat_size, WORKSIZE_1D);
        int rows = cgradients[0].size();
        int cols = cgradients[0][0].size();

        for(int i = 0; i < batchSize; i++) {
            // dL/dC_1 (vecxvec2mat: d_in[i] x d_incoming[0][i])
            vecxvec2mat<<<gridWeightGradFirst, block_1d>>>(
                d_in[i], d_incoming[0][i], d_gradC[0], cweight_rows, cweight_cols
            );
            CU_CHECK(cudaMemcpy(d_totalCgrad + i * cweight_flat_size, d_gradC[0], sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));

            // dL/dB_1 (vecxvec2mat: d_ones x d_incoming[0][i])
            vecxvec2mat<<<gridWeightGradFirst, block_1d>>>(
                d_ones, d_incoming[0][i], d_gradB[0], cweight_rows, cweight_cols
            );
            CU_CHECK(cudaMemcpy(d_totalBgrad + i * cweight_flat_size, d_gradB[0], sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));
        }

        // Average the gradients
        matrix_vector_average<<<calculate_grid_2d(cols, rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), dim3(WORKSIZE_2D_X, WORKSIZE_2D_Y)>>>(
            d_totalCgrad, d_gradC[0], batchSize, rows, cols
        );
        matrix_vector_average<<<calculate_grid_2d(cols, rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), dim3(WORKSIZE_2D_X, WORKSIZE_2D_Y)>>>(
            d_totalBgrad, d_gradB[0], batchSize, rows, cols
        );
        
        // Scale by ALPHA / 1-ALPHA
        scaleByValue<<<gridWeightGradFirst, block_1d>>>(d_gradC[0], d_gradC[0], ALPHA, (int)cweight_flat_size);
        scaleByValue<<<gridWeightGradFirst, block_1d>>>(d_gradB[0], d_gradB[0], 1.0f - ALPHA, (int)cweight_flat_size);

        // --- Update Weights and Copy Results Back ---
        for (int i = 0; i < this->layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            dim3 gridUpdate = calculate_grid_1d(c_size, WORKSIZE_1D);
            dim3 gridUpdateB = calculate_grid_1d(b_size, WORKSIZE_1D);

            switch (weightUpdateType) {
                case 0:
                    kernelUpdateWeights<<<gridUpdate, block_1d>>>(d_cweights[i], d_gradC[i], learningRate, (int)c_size);
                    kernelUpdateWeights<<<gridUpdateB, block_1d>>>(d_bweights[i], d_gradB[i], learningRate, (int)b_size);
                    break;
                case 1:
                    kernelUpdateWeightsWithL1<<<gridUpdate, block_1d>>>(d_cweights[i], d_gradC[i], (int)c_size, learningRate, LAMBDA_L1);
                    kernelUpdateWeightsWithL1<<<gridUpdateB, block_1d>>>(d_bweights[i], d_gradB[i], (int)b_size, learningRate, LAMBDA_L1);
                    break;
                case 2:
                    kernelUpdateWeightsWithL2<<<gridUpdate, block_1d>>>(d_cweights[i], d_gradC[i], (int)c_size, learningRate, LAMBDA_L2);
                    kernelUpdateWeightsWithL2<<<gridUpdateB, block_1d>>>(d_bweights[i], d_gradB[i], (int)b_size, learningRate, LAMBDA_L2);
                    break;
                case 3:
                    kernelUpdateWeightsElasticNet<<<gridUpdate, block_1d>>>(d_cweights[i], d_gradC[i], (int)c_size, learningRate, LAMBDA_L1, LAMBDA_L2);
                    kernelUpdateWeightsElasticNet<<<gridUpdateB, block_1d>>>(d_bweights[i], d_gradB[i], (int)b_size, learningRate, LAMBDA_L1, LAMBDA_L2);
                    break;
                case 4:
                    kernelUpdateWeightsWithWeightDecay<<<gridUpdate, block_1d>>>(d_cweights[i], d_gradC[i], (int)c_size, learningRate, WEIGHT_DECAY);
                    kernelUpdateWeightsWithWeightDecay<<<gridUpdateB, block_1d>>>(d_bweights[i], d_gradB[i], (int)b_size, learningRate, WEIGHT_DECAY);
                    break;
                case 5:
                    kernelUpdateWeightsDropout<<<gridUpdate, block_1d>>>(d_cweights[i], d_gradC[i], (int)c_size, learningRate, DROPOUT_RATE, (unsigned int)rand());
                    kernelUpdateWeightsDropout<<<gridUpdateB, block_1d>>>(d_bweights[i], d_gradB[i], (int)b_size, learningRate, DROPOUT_RATE, (unsigned int)rand());
                    break;
            }

            // Read back updated weights and gradients to host
            std::vector<float> c_upd(c_size), b_upd(b_size), cg_upd(c_size), bg_upd(b_size);
            CU_CHECK(cudaMemcpy(c_upd.data(), d_cweights[i], sizeof(float) * c_size, cudaMemcpyDeviceToHost));
            CU_CHECK(cudaMemcpy(b_upd.data(), d_bweights[i], sizeof(float) * b_size, cudaMemcpyDeviceToHost));
            CU_CHECK(cudaMemcpy(cg_upd.data(), d_gradC[i], sizeof(float) * c_size, cudaMemcpyDeviceToHost));
            CU_CHECK(cudaMemcpy(bg_upd.data(), d_gradB[i], sizeof(float) * b_size, cudaMemcpyDeviceToHost));

            cweights[i] = reshape(c_upd, cweights[i].size(), cweights[i][0].size());
            bweights[i] = reshape(b_upd, bweights[i].size(), bweights[i][0].size());
            cgradients[i] = reshape(cg_upd, cgradients[i].size(), cgradients[i][0].size());
            bgradients[i] = reshape(bg_upd, bgradients[i].size(), bgradients[i][0].size());
        }

        // --- Cleanup ---
        CU_CHECK(cudaFree(d_totalCgrad)); CU_CHECK(cudaFree(d_totalBgrad));
        CU_CHECK(cudaFree(d_preoutgoing_l)); CU_CHECK(cudaFree(d_outgoing_l));
        CU_CHECK(cudaFree(d_dpow_l)); CU_CHECK(cudaFree(d_dact_l));
        CU_CHECK(cudaFree(d_ones));
        for (int i = 0; i < batchSize; i++) {
            CU_CHECK(cudaFree(d_in[i])); CU_CHECK(cudaFree(d_exp[i])); CU_CHECK(cudaFree(d_out[i])); CU_CHECK(cudaFree(d_err[i]));
        }
        for (int i = 0; i < this->layers; i++) {
            CU_CHECK(cudaFree(d_cweights[i])); CU_CHECK(cudaFree(d_bweights[i]));
            CU_CHECK(cudaFree(d_gradC[i])); CU_CHECK(cudaFree(d_gradB[i]));
            for(int j = 0; j < batchSize; j++) {
                CU_CHECK(cudaFree(d_dotProds[i][j])); CU_CHECK(cudaFree(d_activate[i][j]));
                CU_CHECK(cudaFree(d_incoming[i][j]));
            }
        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn1d::cuBackprop (batch): ") + e.what());
    }
}

#endif