#ifdef USE_CU
#include "mnn1d.hpp"
#include "mnn2d.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm> // For std::copy
#include <cmath> // For std::ceil
#include <iostream>

/**
 * @brief batch backpropgation using CUDA for mnn
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
            CU_CHECK(cudaMemcpy(d_out[i], outputBatch[i].data(), sizeof(float) * batch_output_size, cudaMemcpyHostToDevice));
        }

        // Initial Error Calculation
        for (int i = 0; i < batchSize; i++) {
            size_t batch_output_size = outputBatch[i].size();
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

            // Update C weights
            kernelUpdateWeightsElasticNet<<<gridUpdate, block_1d>>>(
                d_cweights[i], d_gradC[i], (int)c_size, learningRate, LAMBDA_L1, LAMBDA_L2
            );

            // Update B weights
            kernelUpdateWeightsElasticNet<<<gridUpdate, block_1d>>>(
                d_bweights[i], d_gradB[i], (int)b_size, learningRate, LAMBDA_L1, LAMBDA_L2
            );

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

// --- mnn2d::cuBackprop Implementation (Batch, 2D input) ---

/**
 * @brief batch backpropgation using CUDA for mnn2d
 * @param expected expected result from forprop
 */
void mnn2d::cuBackprop(const std::vector<std::vector<float>>& expected) {

    try {
        int batchSize = expected.size();
        if (batchSize == 0) {
            throw std::runtime_error("Batch size should be greater than 0.");
        }
 
        dim3 block_1d(WORKSIZE_1D);
        dim3 block_2d(WORKSIZE_2D_X, WORKSIZE_2D_Y);

        // --- Device Memory Pointers (Shared) ---
        std::vector<float*> d_cweights(this->layers), d_bweights(this->layers);
        std::vector<float*> d_gradC(this->layers), d_gradB(this->layers);

        // --- Reusable Temporary Device Buffers ---
        float *d_totalCgrad = nullptr, *d_totalBgrad = nullptr;
        float *d_grad_x_CT = nullptr, *d_dprev_p = nullptr, *d_dprev_act = nullptr; // *d_ones = nullptr;
 
        // --- Device Memory Pointers (Per Batch Element) ---
        std::vector<float*> d_in(batchSize), d_exp(batchSize), d_out(batchSize), d_err(batchSize);
        std::vector<std::vector<float*>> d_incoming(this->layers), d_dotProds(this->layers), d_activate(this->layers);
        for(int i = 0; i < layers; ++i) {
            d_incoming[i].resize(batchSize);
            d_dotProds[i].resize(batchSize);
            d_activate[i].resize(batchSize);
        }
 
        // --- Allocate and Copy Per-Batch Data ---
        for (int i = 0; i < batchSize; i++) {
            size_t in_size = inputBatch[i].size() * inputBatch[i][0].size();
            size_t out_size = outputBatch[i].size();
            CU_CHECK(cudaMalloc(&d_in[i], sizeof(float) * in_size));
            CU_CHECK(cudaMalloc(&d_exp[i], sizeof(float) * out_size));
            CU_CHECK(cudaMalloc(&d_out[i], sizeof(float) * out_size));
            CU_CHECK(cudaMalloc(&d_err[i], sizeof(float) * out_size));
            CU_CHECK(cudaMemcpy(d_in[i], flatten(inputBatch[i]).data(), sizeof(float) * in_size, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(d_exp[i], expected[i].data(), sizeof(float) * out_size, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(d_out[i], outputBatch[i].data(), sizeof(float) * out_size, cudaMemcpyHostToDevice));
 
            subtract<<<calculate_grid_1d(out_size, WORKSIZE_1D), block_1d>>>(d_out[i], d_exp[i], d_err[i], (int)out_size);
        }
 
        // --- Allocate and Copy Shared Data & Average Gradient Buffers ---
        size_t max_total_grad_size = 0;
        size_t max_act_size = 0;
        for(int i = 0; i < this->layers; i++) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            max_total_grad_size = std::max(max_total_grad_size, c_size);
 
            CU_CHECK(cudaMalloc(&d_cweights[i], sizeof(float) * c_size));
            CU_CHECK(cudaMalloc(&d_bweights[i], sizeof(float) * b_size));
            CU_CHECK(cudaMalloc(&d_gradC[i], sizeof(float) * c_size));
            CU_CHECK(cudaMalloc(&d_gradB[i], sizeof(float) * b_size));
            CU_CHECK(cudaMemcpy(d_cweights[i], flatten(cweights[i]).data(), sizeof(float) * c_size, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(d_bweights[i], flatten(bweights[i]).data(), sizeof(float) * b_size, cudaMemcpyHostToDevice));
 
            for(int j = 0; j < batchSize; j++) {
                size_t act_size = actBatch[i][j].size() * actBatch[i][j][0].size();
                size_t dot_size = dotBatch[i][j].size() * dotBatch[i][j][0].size();
                max_act_size = std::max(max_act_size, act_size);
                CU_CHECK(cudaMalloc(&d_activate[i][j], sizeof(float) * act_size));
                CU_CHECK(cudaMalloc(&d_dotProds[i][j], sizeof(float) * dot_size));
                CU_CHECK(cudaMalloc(&d_incoming[i][j], sizeof(float) * act_size));
                CU_CHECK(cudaMemcpy(d_activate[i][j], flatten(actBatch[i][j]).data(), sizeof(float) * act_size, cudaMemcpyHostToDevice));
                CU_CHECK(cudaMemcpy(d_dotProds[i][j], flatten(dotBatch[i][j]).data(), sizeof(float) * dot_size, cudaMemcpyHostToDevice));
            }
        }
 
        // Allocate buffers for average gradient accumulation
        CU_CHECK(cudaMalloc(&d_totalCgrad, max_total_grad_size * batchSize * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_totalBgrad, max_total_grad_size * batchSize * sizeof(float)));
        // Allocate reusable intermediate buffers for outgoing gradient calculation
        CU_CHECK(cudaMalloc(&d_grad_x_CT, sizeof(float) * max_act_size));
        CU_CHECK(cudaMalloc(&d_dprev_p, sizeof(float) * max_act_size));
        CU_CHECK(cudaMalloc(&d_dprev_act, sizeof(float) * max_act_size));
 
        // --- Initial Error Calculation (Output Layer) ---
        // Backpropagate error through mean pooling layer for each batch item
        for (int i = 0; i < batchSize; i++) {
            size_t out_size = outputBatch[i].size();
            subtract<<<calculate_grid_1d(out_size, WORKSIZE_1D), block_1d>>>(d_out[i], d_exp[i], d_err[i], (int)out_size);

            // Distribute the error back through the mean pooling layer
            size_t last_layer_rows = actBatch[layers-1][i].size();
            for(size_t r = 0; r < last_layer_rows; ++r) {
                CU_CHECK(cudaMemcpy(d_incoming[layers - 1][i] + (r * outWidth), d_err[i], sizeof(float) * outWidth, cudaMemcpyDeviceToDevice));
            }
        }
 
        // --- Backpropagation Loop (Last Layer to Second Layer) ---
        for(int layer = layers - 1; layer >= 1; --layer) {
            int cweight_rows = cweights[layer].size();
            int cweight_cols = cweights[layer][0].size();
            size_t cweight_flat_size = cweight_rows * cweight_cols;

            // Allocate temporary buffers for this layer's calculations (reused for each batch item)
            float *d_C_T = nullptr, *d_prev_p = nullptr, *d_prev_p_T = nullptr, *d_onesT = nullptr;
            size_t prev_act_flat_size_max = actBatch[layer-1][0].size() * actBatch[layer-1][0][0].size();
            CU_CHECK(cudaMalloc(&d_C_T, sizeof(float) * cweight_flat_size));
 
            for(int i = 0; i < batchSize; ++i) {
                int prev_rows = actBatch[layer-1][i].size();
                int prev_cols = actBatch[layer-1][i][0].size();
                int curr_rows = actBatch[layer][i].size();
                int curr_cols = actBatch[layer][i][0].size();
                size_t prev_act_flat_size = prev_rows * prev_cols;
 
                // --- 1. Calculate Outgoing Gradient for Previous Layer (d_incoming[layer-1]) ---
                // Transpose C
                transpose<<<calculate_grid_2d(cweight_cols, cweight_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                    d_cweights[layer], d_C_T, cweight_rows, cweight_cols
                );
                // Calculate dL/dz_l * C^T
                matxmat2mat<<<calculate_grid_2d(prev_cols, curr_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                    d_incoming[layer][i], d_C_T, d_grad_x_CT, curr_rows, cweight_cols, cweight_rows
                );
 
                // d(prev_p)
                dPower<<<calculate_grid_1d(prev_act_flat_size, WORKSIZE_1D), block_1d>>>(
                    d_activate[layer-1][i], d_dprev_p, order, prev_act_flat_size
                );
                
                // d(prev_act) (Softmax derivative)
                size_t prev_dot_size = dotBatch[layer-1][i].size() * dotBatch[layer-1][i][0].size();
                softmaxDer<<<calculate_grid_1d(prev_dot_size, WORKSIZE_1D), block_1d>>>(
                    d_dotProds[layer-1][i], d_dprev_act, SOFTMAX_TEMP, (int)prev_dot_size
                );
 
                // outgoing gradient = (dL/dz_l * C^T) .* d(prev_p) .* d(prev_act)
                hadamard2<<<calculate_grid_1d(prev_act_flat_size, WORKSIZE_1D), block_1d>>>(
                    d_grad_x_CT, d_dprev_p, d_dprev_act, d_incoming[layer-1][i], prev_rows, prev_cols
                );
 
                // --- 2. Calculate Weight Gradients for Current Layer (d_gradC, d_gradB) ---
                // dL/dC_layer = prev_p^T * dL/dz_l
                CU_CHECK(cudaMalloc(&d_prev_p, sizeof(float) * prev_act_flat_size));
                power<<<calculate_grid_1d(prev_act_flat_size, WORKSIZE_1D), block_1d>>>(
                    d_activate[layer-1][i], d_prev_p, order, prev_act_flat_size
                );
                CU_CHECK(cudaMalloc(&d_prev_p_T, sizeof(float) * prev_act_flat_size));
                transpose<<<calculate_grid_2d(prev_cols, prev_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                    d_prev_p, d_prev_p_T, prev_rows, prev_cols
                );
                matxmat2mat<<<calculate_grid_2d(cweight_cols, prev_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                    d_prev_p_T, d_incoming[layer][i], d_gradC[layer], prev_cols, prev_rows, cweight_cols
                );
                CU_CHECK(cudaMemcpy(d_totalCgrad + i * cweight_flat_size, d_gradC[layer], sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));
                CU_CHECK(cudaFree(d_prev_p));
                CU_CHECK(cudaFree(d_prev_p_T));
 
                // dL/dB_layer = ones^T * dL/dz_l
                CU_CHECK(cudaMalloc(&d_onesT, sizeof(float) * prev_act_flat_size));
                std::vector<float> h_ones(prev_act_flat_size, 1.0f); // Host vector of ones
                CU_CHECK(cudaMemcpy(d_onesT, h_ones.data(), sizeof(float) * prev_act_flat_size, cudaMemcpyHostToDevice));
                // transpose<<<calculate_grid_2d(prev_cols, prev_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                //     d_ones, d_onesT, prev_rows, prev_cols
                // );
                matxmat2mat<<<calculate_grid_2d(cweight_cols, prev_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                    d_onesT, d_incoming[layer][i], d_gradB[layer], prev_cols, prev_rows, cweight_cols
                );
                CU_CHECK(cudaMemcpy(d_totalBgrad + i * cweight_flat_size, d_gradB[layer], sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));
                CU_CHECK(cudaFree(d_onesT));
            }
            // --- 3. Average Gradients and Scale ---
            matrix_vector_average<<<calculate_grid_2d(cweight_cols, cweight_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                d_totalCgrad, d_gradC[layer], batchSize, cweight_rows, cweight_cols
            );
            matrix_vector_average<<<calculate_grid_2d(cweight_cols, cweight_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                d_totalBgrad, d_gradB[layer], batchSize, cweight_rows, cweight_cols
            );
            scaleByValue<<<calculate_grid_1d(cweight_flat_size, WORKSIZE_1D), block_1d>>>(
                d_gradC[layer], d_gradC[layer], ALPHA, (int)cweight_flat_size
            );
            scaleByValue<<<calculate_grid_1d(cweight_flat_size, WORKSIZE_1D), block_1d>>>(
                d_gradB[layer], d_gradB[layer], (1.0f - ALPHA), (int)cweight_flat_size
            );
            CU_CHECK(cudaFree(d_C_T));
        }
 
        // --- Backpropagation for the First Layer (Layer 0) ---
        int cweight_rows_first = cweights[0].size();
        int cweight_cols_first = cweights[0][0].size();
        size_t cweight_flat_size_first = cweight_rows_first * cweight_cols_first;
        float *d_input_p = nullptr, *d_input_p_T = nullptr, *d_onesT = nullptr;

        for(int i = 0; i < batchSize; i++) {
            int inHeight = inputBatch[i].size();
            int inWidth = inputBatch[i][0].size();
            size_t input_flat_size = inHeight * inWidth;
 
            // Calculate d_input_p = input^order
            CU_CHECK(cudaMalloc(&d_input_p, sizeof(float) * input_flat_size));
            power<<<calculate_grid_1d(input_flat_size, WORKSIZE_1D), block_1d>>>(
                d_in[i], d_input_p, order, input_flat_size
            );
            // Transpose d_input_p
            CU_CHECK(cudaMalloc(&d_input_p_T, sizeof(float) * input_flat_size));
            transpose<<<calculate_grid_2d(inWidth, inHeight, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                d_input_p, d_input_p_T, inHeight, inWidth
            );
            // Calculate dL/dC_0 = d_input_p_T * dL/dz_0
            matxmat2mat<<<calculate_grid_2d(cweight_cols_first, inHeight, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                d_input_p_T, d_incoming[0][i], d_gradC[0], inWidth, inHeight, cweight_cols_first
            );
            CU_CHECK(cudaMemcpy(d_totalCgrad + i * cweight_flat_size_first, d_gradC[0], sizeof(float) * cweight_flat_size_first, cudaMemcpyDeviceToDevice));
            CU_CHECK(cudaFree(d_input_p)); CU_CHECK(cudaFree(d_input_p_T));
 
            // Calculate dL/dB_0 = ones^T * dL/dz_0
            std::vector<float> h_ones(input_flat_size, 1.0f); // Host vector of ones
            CU_CHECK(cudaMalloc(&d_onesT, sizeof(float) * input_flat_size));
            CU_CHECK(cudaMemcpy(d_onesT, h_ones.data(), sizeof(float) * input_flat_size, cudaMemcpyHostToDevice));
            // transpose<<<calculate_grid_2d(inWidth, inHeight, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
            //     d_ones, d_onesT, inHeight, inWidth
            // );
            matxmat2mat<<<calculate_grid_2d(cweight_cols_first, inHeight, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                d_onesT, d_incoming[0][i], d_gradB[0], inWidth, inHeight, cweight_cols_first
            );
            CU_CHECK(cudaMemcpy(d_totalBgrad + i * cweight_flat_size_first, d_gradB[0], sizeof(float) * cweight_flat_size_first, cudaMemcpyDeviceToDevice));
            CU_CHECK(cudaFree(d_onesT));
        }

        // Average the gradients
        matrix_vector_average<<<calculate_grid_2d(cweight_cols_first, cweight_rows_first, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
            d_totalCgrad, d_gradC[0], batchSize, cweight_rows_first, cweight_cols_first
        );
        matrix_vector_average<<<calculate_grid_2d(cweight_cols_first, cweight_rows_first, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
            d_totalBgrad, d_gradB[0], batchSize, cweight_rows_first, cweight_cols_first
        );
        // Scale by value
        scaleByValue<<<calculate_grid_1d(cweight_flat_size_first, WORKSIZE_1D), block_1d>>>(d_gradC[0], d_gradC[0], ALPHA, (int)cweight_flat_size_first);
        scaleByValue<<<calculate_grid_1d(cweight_flat_size_first, WORKSIZE_1D), block_1d>>>(d_gradB[0], d_gradB[0], (1.0f - ALPHA), (int)cweight_flat_size_first);
 
        // --- Update Weights and Copy Results Back --- (Same as mnn)
        for (int i = 0; i < this->layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            dim3 gridUpdate = calculate_grid_1d(c_size, WORKSIZE_1D);

            kernelUpdateWeightsElasticNet<<<gridUpdate, block_1d>>>(
                d_cweights[i], d_gradC[i], (int)c_size, learningRate, LAMBDA_L1, LAMBDA_L2
            );
            kernelUpdateWeightsElasticNet<<<gridUpdate, block_1d>>>(
                d_bweights[i], d_gradB[i], (int)b_size, learningRate, LAMBDA_L1, LAMBDA_L2
            );

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
        CU_CHECK(cudaFree(d_grad_x_CT)); CU_CHECK(cudaFree(d_dprev_p)); CU_CHECK(cudaFree(d_dprev_act));
        // CU_CHECK(cudaFree(d_ones));
        for (int i = 0; i < batchSize; ++i) {
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
        throw std::runtime_error(std::string("Exception in mnn2d::cuBackprop (batch): ") + e.what());
    }
}

#endif