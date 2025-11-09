#ifdef USE_CUDA
#include "mnn.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

/**
 * @brief batch backpropgation using CUDA for mnn
 * @param expected expected result from forprop
 */
void mnn::cuBackprop(const std::vector<std::vector<float>>& expected) {
    try {
        int batchSize = expected.size();
        if (batchSize == 0) {
            throw std::runtime_error("Batch size should be greater than 0.");
        }

        dim3 block_1d(CUDA_BLOCK_SIZE_1D);
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

            CUDA_CHECK(cudaMalloc(&d_in[i], sizeof(float) * batch_input_size));
            CUDA_CHECK(cudaMalloc(&d_exp[i], sizeof(float) * batch_output_size));
            CUDA_CHECK(cudaMalloc(&d_out[i], sizeof(float) * batch_output_size));
            CUDA_CHECK(cudaMalloc(&d_err[i], sizeof(float) * batch_output_size));

            CUDA_CHECK(cudaMemcpy(d_in[i], inputBatch[i].data(), sizeof(float) * batch_input_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_exp[i], expected[i].data(), sizeof(float) * batch_output_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_out[i], outputBatch[i].data(), sizeof(float) * batch_output_size, cudaMemcpyHostToDevice));

            // Initial Error Calculation
            dim3 gridSub = calculate_grid_1d(batch_output_size, CUDA_BLOCK_SIZE_1D);
            subtract<<<gridSub, block_1d>>>(d_out[i], d_exp[i], d_err[i], (int)batch_output_size);
        }

        // --- Allocate and Copy Shared Data & Batch Activations ---
        size_t max_total_grad_size = 0;
        for(int i = 0; i < this->layers; i++) {
            std::vector<float> flat_c = flatten(cweights[i]);
            std::vector<float> flat_b = flatten(bweights[i]);
            size_t c_size = flat_c.size();
            max_total_grad_size = std::max(max_total_grad_size, c_size);

            CUDA_CHECK(cudaMalloc(&d_cweights[i], sizeof(float) * c_size));
            CUDA_CHECK(cudaMalloc(&d_bweights[i], sizeof(float) * flat_b.size()));
            CUDA_CHECK(cudaMalloc(&d_gradC[i], sizeof(float) * c_size));
            CUDA_CHECK(cudaMalloc(&d_gradB[i], sizeof(float) * flat_b.size()));
            CUDA_CHECK(cudaMemcpy(d_cweights[i], flat_c.data(), sizeof(float) * c_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_bweights[i], flat_b.data(), sizeof(float) * flat_b.size(), cudaMemcpyHostToDevice));

            for(int j = 0; j < batchSize; j++) {
                size_t act_size = actBatch[i][j].size();
                CUDA_CHECK(cudaMalloc(&d_activate[i][j], sizeof(float) * act_size));
                CUDA_CHECK(cudaMalloc(&d_dotProds[i][j], sizeof(float) * act_size));
                CUDA_CHECK(cudaMalloc(&d_incoming[i][j], sizeof(float) * act_size));
                CUDA_CHECK(cudaMemcpy(d_activate[i][j], actBatch[i][j].data(), sizeof(float) * act_size, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_dotProds[i][j], dotBatch[i][j].data(), sizeof(float) * act_size, cudaMemcpyHostToDevice));
                if (i == layers - 1) { // Copy initial error to last layer's incoming gradient
                    CUDA_CHECK(cudaMemcpy(d_incoming[i][j], d_err[j], sizeof(float) * act_size, cudaMemcpyDeviceToDevice));
                }
            }
        }

        // Allocate buffers for average gradient accumulation (Total size = max_total_grad_size * batchSize)
        float *d_totalCgrad = nullptr, *d_totalBgrad = nullptr;
        CUDA_CHECK(cudaMalloc(&d_totalCgrad, max_total_grad_size * batchSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_totalBgrad, max_total_grad_size * batchSize * sizeof(float)));

        // Allocate shared intermediate buffers for outgoing gradient calculation
        float *d_preoutgoing_l = nullptr, *d_outgoing_l = nullptr;
        float *d_dpow_l = nullptr, *d_dact_l = nullptr;
        CUDA_CHECK(cudaMalloc(&d_preoutgoing_l, max_layer_width * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_outgoing_l, max_layer_width * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dpow_l, max_layer_width * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dact_l, max_layer_width * sizeof(float)));

        // Allocate and copy ones vector
        std::vector<float> v1(max_layer_width, 1.0f);
        CUDA_CHECK(cudaMalloc(&d_ones, sizeof(float) * max_layer_width));
        CUDA_CHECK(cudaMemcpy(d_ones, v1.data(), sizeof(float) * max_layer_width, cudaMemcpyHostToDevice));


        // --- Backpropagation Loop (Last Layer to Second Layer) ---
        for(int layer = layers - 1; layer >= 1; --layer) {
            int prev_size = actBatch[layer - 1][0].size();
            int curr_size = actBatch[layer][0].size();
            int cweight_rows = prev_size;
            int cweight_cols = curr_size;
            size_t cweight_flat_size = cweight_rows * cweight_cols;
            dim3 gridWeightGrad = calculate_grid_1d(cweight_flat_size, CUDA_BLOCK_SIZE_1D);
            dim3 gridOutGrad = calculate_grid_1d(prev_size, CUDA_BLOCK_SIZE_1D);

            for(int i = 0; i < batchSize; ++i) {
                // 1. dL/dC_l (vecxvec2mat: d_activate[L-1][i] x d_incoming[L][i])
                vecxvec2mat<<<gridWeightGrad, block_1d>>>(
                    d_activate[layer - 1][i], d_incoming[layer][i], d_gradC[layer], cweight_rows, cweight_cols
                );
                // Copy to total C grad buffer
                CUDA_CHECK(cudaMemcpy(d_totalCgrad + i * cweight_flat_size, d_gradC[layer], sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));

                // 2. dL/dB_l (vecxvec2mat: d_ones x d_incoming[L][i])
                vecxvec2mat<<<gridWeightGrad, block_1d>>>(
                    d_ones, d_incoming[layer][i], d_gradB[layer], cweight_rows, cweight_cols
                );
                // Copy to total B grad buffer
                CUDA_CHECK(cudaMemcpy(d_totalBgrad + i * cweight_flat_size, d_gradB[layer], sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));

                // 3. Outgoing Gradient Calculation
                float *d_C_T = nullptr;
                CUDA_CHECK(cudaMalloc(&d_C_T, sizeof(float) * cweight_flat_size));
                transpose<<<calculate_grid_2d(cweight_cols, cweight_rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), dim3(CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y)>>>(
                    d_cweights[layer], d_C_T, cweight_rows, cweight_cols
                );
                // incoming gradient x C^T (vecxmat2vec)
                vecxmat2vec<<<gridOutGrad, block_1d>>>(
                    d_incoming[layer][i], d_C_T, d_preoutgoing_l, cweight_cols, cweight_rows
                );
                CUDA_CHECK(cudaFree(d_C_T));

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
            matrix_vector_average<<<calculate_grid_2d(cols, rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), dim3(CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y)>>>(
                d_totalCgrad, d_gradC[layer], batchSize, rows, cols
            );
            matrix_vector_average<<<calculate_grid_2d(cols, rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), dim3(CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y)>>>(
                d_totalBgrad, d_gradB[layer], batchSize, rows, cols
            );
            
            // Scale by alpha / 1-alpha (done during averaging/update in CL, but here it's separate)
            scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradC[layer], d_gradC[layer], alpha, (int)cweight_flat_size);
            scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradB[layer], d_gradB[layer], alpha, (int)cweight_flat_size); // CL uses alpha for both

            // Weight update is done after the loop.
        }

        // --- Backpropagation for the First Layer (Layer 0) ---
        int cweight_rows = inputSize;
        int cweight_cols = cweights[0][0].size();
        size_t cweight_flat_size = cweight_rows * cweight_cols;
        dim3 gridWeightGradFirst = calculate_grid_1d(cweight_flat_size, CUDA_BLOCK_SIZE_1D);
        int rows = cgradients[0].size();
        int cols = cgradients[0][0].size();

        for(int i = 0; i < batchSize; i++) {
            // dL/dC_1 (vecxvec2mat: d_in[i] x d_incoming[0][i])
            vecxvec2mat<<<gridWeightGradFirst, block_1d>>>(
                d_in[i], d_incoming[0][i], d_gradC[0], cweight_rows, cweight_cols
            );
            CUDA_CHECK(cudaMemcpy(d_totalCgrad + i * cweight_flat_size, d_gradC[0], sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));

            // dL/dB_1 (vecxvec2mat: d_ones x d_incoming[0][i])
            vecxvec2mat<<<gridWeightGradFirst, block_1d>>>(
                d_ones, d_incoming[0][i], d_gradB[0], cweight_rows, cweight_cols
            );
            CUDA_CHECK(cudaMemcpy(d_totalBgrad + i * cweight_flat_size, d_gradB[0], sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));
        }

        // Average the gradients
        matrix_vector_average<<<calculate_grid_2d(cols, rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), dim3(CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y)>>>(
            d_totalCgrad, d_gradC[0], batchSize, rows, cols
        );
        matrix_vector_average<<<calculate_grid_2d(cols, rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), dim3(CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y)>>>(
            d_totalBgrad, d_gradB[0], batchSize, rows, cols
        );
        
        // Scale by alpha / 1-alpha
        scaleByValue<<<gridWeightGradFirst, block_1d>>>(d_gradC[0], d_gradC[0], alpha, (int)cweight_flat_size);
        scaleByValue<<<gridWeightGradFirst, block_1d>>>(d_gradB[0], d_gradB[0], alpha, (int)cweight_flat_size); // CL uses alpha for both


        // --- Update Weights and Copy Results Back ---
        for (int i = 0; i < this->layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            dim3 gridUpdate = calculate_grid_1d(c_size, CUDA_BLOCK_SIZE_1D);

            // Update C weights
            kernelUpdateWeightsElasticNet<<<gridUpdate, block_1d>>>(d_cweights[i], d_gradC[i], (int)c_size, learningRate, LAMBDA_L1, LAMBDA_L2);

            // Update B weights
            kernelUpdateWeightsElasticNet<<<gridUpdate, block_1d>>>(d_bweights[i], d_gradB[i], (int)b_size, learningRate, LAMBDA_L1, LAMBDA_L2);

            // Read back updated weights and gradients to host
            std::vector<float> c_upd(c_size), b_upd(b_size), cg_upd(c_size), bg_upd(b_size);
            CUDA_CHECK(cudaMemcpy(c_upd.data(), d_cweights[i], sizeof(float) * c_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(b_upd.data(), d_bweights[i], sizeof(float) * b_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(cg_upd.data(), d_gradC[i], sizeof(float) * c_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(bg_upd.data(), d_gradB[i], sizeof(float) * b_size, cudaMemcpyDeviceToHost));

            cweights[i] = reshape(c_upd, cweights[i].size(), cweights[i][0].size());
            bweights[i] = reshape(b_upd, bweights[i].size(), bweights[i][0].size());
            cgradients[i] = reshape(cg_upd, cgradients[i].size(), cgradients[i][0].size());
            bgradients[i] = reshape(bg_upd, bgradients[i].size(), bgradients[i][0].size());
        }

        // --- Cleanup ---
        CUDA_CHECK(cudaFree(d_totalCgrad)); CUDA_CHECK(cudaFree(d_totalBgrad));
        CUDA_CHECK(cudaFree(d_preoutgoing_l)); CUDA_CHECK(cudaFree(d_outgoing_l));
        CUDA_CHECK(cudaFree(d_dpow_l)); CUDA_CHECK(cudaFree(d_dact_l));
        CUDA_CHECK(cudaFree(d_ones));
        for (int i = 0; i < batchSize; i++) {
            CUDA_CHECK(cudaFree(d_in[i])); CUDA_CHECK(cudaFree(d_exp[i])); CUDA_CHECK(cudaFree(d_out[i])); CUDA_CHECK(cudaFree(d_err[i]));
        }
        for (int i = 0; i < this->layers; i++) {
            CUDA_CHECK(cudaFree(d_cweights[i])); CUDA_CHECK(cudaFree(d_bweights[i]));
            CUDA_CHECK(cudaFree(d_gradC[i])); CUDA_CHECK(cudaFree(d_gradB[i]));
            for(int j = 0; j < batchSize; j++) {
                CUDA_CHECK(cudaFree(d_dotProds[i][j])); CUDA_CHECK(cudaFree(d_activate[i][j]));
                CUDA_CHECK(cudaFree(d_incoming[i][j]));
            }
        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn::cuBackprop (batch): ") + e.what());
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

        dim3 block_1d(CUDA_BLOCK_SIZE_1D);
        dim3 block_2d(CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y);
        int inHeight = inputBatch[0].size();
        int inWidth = inputBatch[0][0].size();
        size_t outputSize = outputBatch[0].size();

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
        float *d_grad_x_CT = nullptr, *d_dprev_p = nullptr, *d_dprev_act = nullptr;

        // --- Allocate and Copy Per-Batch Data ---
        for (int i = 0; i < batchSize; i++) {
            size_t batch_input_size = inHeight * inWidth;
            size_t batch_output_size = outputBatch[i].size();

            CUDA_CHECK(cudaMalloc(&d_in[i], sizeof(float) * batch_input_size));
            CUDA_CHECK(cudaMalloc(&d_exp[i], sizeof(float) * batch_output_size));
            CUDA_CHECK(cudaMalloc(&d_out[i], sizeof(float) * batch_output_size));
            CUDA_CHECK(cudaMalloc(&d_err[i], sizeof(float) * batch_output_size));

            CUDA_CHECK(cudaMemcpy(d_in[i], flatten(inputBatch[i]).data(), sizeof(float) * batch_input_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_exp[i], expected[i].data(), sizeof(float) * batch_output_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_out[i], outputBatch[i].data(), sizeof(float) * batch_output_size, cudaMemcpyHostToDevice));

            // Initial Error Calculation (output - expected)
            dim3 gridSub = calculate_grid_1d(batch_output_size, CUDA_BLOCK_SIZE_1D);
            subtract<<<gridSub, block_1d>>>(d_out[i], d_exp[i], d_err[i], (int)batch_output_size);

            // Error back through mean pooling layer (Host-side implementation)
            std::vector<float> out_err_host(batch_output_size);
            CUDA_CHECK(cudaMemcpy(out_err_host.data(), d_err[i], sizeof(float) * batch_output_size, cudaMemcpyDeviceToHost));
            
            int last_rows = actBatch[layers-1][i].size();
            int last_cols = actBatch[layers-1][i][0].size();
            std::vector<float> last_layer_err(last_rows * last_cols);
            for(int r = 0; r < last_rows; ++r) {
                for(int c = 0; c < last_cols; ++c) {
                    last_layer_err[r * last_cols + c] = out_err_host[c] / (float)last_rows;
                }
            }
            // Allocate activation buffers and copy data
            for(int l = 0; l < layers; ++l) {
                size_t act_size = actBatch[l][i].size() * actBatch[l][i][0].size();
                CUDA_CHECK(cudaMalloc(&d_activate[l][i], sizeof(float) * act_size));
                CUDA_CHECK(cudaMalloc(&d_dotProds[l][i], sizeof(float) * act_size));
                CUDA_CHECK(cudaMalloc(&d_incoming[l][i], sizeof(float) * act_size));
                CUDA_CHECK(cudaMemcpy(d_activate[l][i], flatten(actBatch[l][i]).data(), sizeof(float) * act_size, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_dotProds[l][i], flatten(dotBatch[l][i]).data(), sizeof(float) * act_size, cudaMemcpyHostToDevice));
            }
            CUDA_CHECK(cudaMemcpy(d_incoming[layers - 1][i], last_layer_err.data(), last_layer_err.size() * sizeof(float), cudaMemcpyHostToDevice));
        }

        // --- Allocate and Copy Shared Data & Average Gradient Buffers ---
        size_t max_total_grad_size = 0;
        for(int i = 0; i < this->layers; i++) {
            std::vector<float> flat_c = flatten(cweights[i]);
            std::vector<float> flat_b = flatten(bweights[i]);
            size_t c_size = flat_c.size();
            max_total_grad_size = std::max(max_total_grad_size, c_size);

            CUDA_CHECK(cudaMalloc(&d_cweights[i], sizeof(float) * c_size));
            CUDA_CHECK(cudaMalloc(&d_bweights[i], sizeof(float) * flat_b.size()));
            CUDA_CHECK(cudaMalloc(&d_gradC[i], sizeof(float) * c_size));
            CUDA_CHECK(cudaMalloc(&d_gradB[i], sizeof(float) * flat_b.size()));
            CUDA_CHECK(cudaMemcpy(d_cweights[i], flat_c.data(), sizeof(float) * c_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_bweights[i], flat_b.data(), sizeof(float) * flat_b.size(), cudaMemcpyHostToDevice));
        }

        // Allocate buffers for average gradient accumulation
        float *d_totalCgrad = nullptr, *d_totalBgrad = nullptr;
        CUDA_CHECK(cudaMalloc(&d_totalCgrad, max_total_grad_size * batchSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_totalBgrad, max_total_grad_size * batchSize * sizeof(float)));

        // Allocate shared intermediate buffers for outgoing gradient calculation
        int max_act_rows = 0, max_act_cols = 0;
        for(int l = 0; l < layers; ++l) {
            max_act_rows = std::max(max_act_rows, (int)actBatch[l][0].size());
            max_act_cols = std::max(max_act_cols, (int)actBatch[l][0][0].size());
        }
        size_t max_act_size = max_act_rows * max_act_cols;
        CUDA_CHECK(cudaMalloc(&d_grad_x_CT, max_act_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dprev_p, max_act_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dprev_act, max_act_size * sizeof(float)));

        // --- Backpropagation Loop (Last Layer to Second Layer) ---
        for(int layer = layers - 1; layer >= 1; layer--) {
            int prev_rows = actBatch[layer-1][0].size();
            int prev_cols = actBatch[layer-1][0][0].size();
            int curr_rows = actBatch[layer][0].size();
            int curr_cols = actBatch[layer][0][0].size();
            int cweight_rows = prev_cols;
            int cweight_cols = curr_cols;
            size_t cweight_flat_size = cweight_rows * cweight_cols;
            dim3 gridWeightGrad = calculate_grid_1d(cweight_flat_size, CUDA_BLOCK_SIZE_1D);
            dim3 gridMatMul = calculate_grid_2d(prev_cols, curr_rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y);
            dim3 gridMatMulGrad = calculate_grid_2d(cweight_cols, cweight_rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y);

            for(int i = 0; i < batchSize; i++) {
                size_t prev_act_size = prev_rows * prev_cols;
                dim3 grid1D_prev = calculate_grid_1d(prev_act_size, CUDA_BLOCK_SIZE_1D);

                // 1. Transpose C: C_P_C -> C_C_P
                float *d_C_T = nullptr;
                CUDA_CHECK(cudaMalloc(&d_C_T, sizeof(float) * cweight_flat_size));
                transpose<<<calculate_grid_2d(cweight_cols, cweight_rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), block_2d>>>(
                    d_cweights[layer], d_C_T, cweight_rows, cweight_cols
                );

                // 2. grad x C^T: (R_curr x C_curr) * (C_curr x P_cols) -> R_curr x P_cols
                matxmat2mat<<<calculate_grid_2d(prev_cols, curr_rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), block_2d>>>(
                    d_incoming[layer][i], d_C_T, d_grad_x_CT, curr_rows, curr_cols, prev_cols
                );
                CUDA_CHECK(cudaFree(d_C_T));

                // 3. d(prev_p)
                dPower<<<grid1D_prev, block_1d>>>(d_activate[layer-1][i], d_dprev_p, order, (int)prev_act_size);

                // 4. d(prev_act) (Softmax Derivative)
                size_t prev_dot_size = prev_act_size;
                if (prev_dot_size > 256) throw std::runtime_error("Softmax derivative size too large.");
                softmaxDer<<<dim3(1), dim3(prev_dot_size)>>>(d_dotProds[layer-1][i], d_dprev_act, SOFTMAX_TEMP, (int)prev_dot_size);

                // 5. outgoing = (dL/dz * C^T) . d(prev_p) . d(prev_act)
                hadamard2<<<grid1D_prev, block_1d>>>(
                    d_grad_x_CT, d_dprev_p, d_dprev_act, d_incoming[layer-1][i], prev_rows, prev_cols // d_outgoing[layer-1][i] -> copied to d_incoming[layer-1][i]
                );

                // Power prev_act: a_prev -> a_prev^n
                float *d_prev_p_i = nullptr;
                CUDA_CHECK(cudaMalloc(&d_prev_p_i, sizeof(float) * prev_act_size));
                power<<<grid1D_prev, block_1d>>>(d_activate[layer-1][i], d_prev_p_i, order, (int)prev_act_size);
                
                // 7. Transpose prev_p: (R_prev x P_cols) -> (P_cols x R_prev)
                float *d_prev_p_T_i = nullptr;
                size_t prev_p_T_size = prev_cols * prev_rows;
                CUDA_CHECK(cudaMalloc(&d_prev_p_T_i, sizeof(float) * prev_p_T_size));
                transpose<<<calculate_grid_2d(prev_rows, prev_cols, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), block_2d>>>(
                    d_prev_p_i, d_prev_p_T_i, prev_rows, prev_cols
                );
                CUDA_CHECK(cudaFree(d_prev_p_i));

                // 8. dL/dC_layer: (P_cols x R_curr) * (R_curr x C_curr) -> P_cols x C_curr
                float *d_gradC_i = nullptr; CUDA_CHECK(cudaMalloc(&d_gradC_i, sizeof(float) * cweight_flat_size));
                matxmat2mat<<<gridMatMulGrad, block_2d>>>(d_prev_p_T_i, d_incoming[layer][i], d_gradC_i, cweight_rows, curr_rows, cweight_cols);
                scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradC_i, d_gradC_i, alpha, (int)cweight_flat_size);
                CUDA_CHECK(cudaMemcpy(d_totalCgrad + i * cweight_flat_size, d_gradC_i, sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaFree(d_gradC_i));

                // 9. dL/dB_layer
                float *d_ones_i = nullptr; CUDA_CHECK(cudaMalloc(&d_ones_i, sizeof(float) * prev_act_size));
                float *d_onesT_i = nullptr; CUDA_CHECK(cudaMalloc(&d_onesT_i, sizeof(float) * prev_p_T_size));
                std::vector<float> ones(prev_act_size, 1.0f); CUDA_CHECK(cudaMemcpy(d_ones_i, ones.data(), sizeof(float) * prev_act_size, cudaMemcpyHostToDevice));
                transpose<<<calculate_grid_2d(prev_cols, prev_rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), block_2d>>>(d_ones_i, d_onesT_i, prev_rows, prev_cols);
                CUDA_CHECK(cudaFree(d_ones_i));
                float *d_gradB_i = nullptr; CUDA_CHECK(cudaMalloc(&d_gradB_i, sizeof(float) * cweight_flat_size));
                matxmat2mat<<<gridMatMulGrad, block_2d>>>(d_onesT_i, d_incoming[layer][i], d_gradB_i, cweight_rows, curr_rows, cweight_cols);
                scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradB_i, d_gradB_i, 1.0f - alpha, (int)cweight_flat_size);
                CUDA_CHECK(cudaMemcpy(d_totalBgrad + i * cweight_flat_size, d_gradB_i, sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaFree(d_gradB_i));
                CUDA_CHECK(cudaFree(d_onesT_i));
                CUDA_CHECK(cudaFree(d_prev_p_T_i));
            }

            // 10. Average the Gradients
            int rows = cgradients[layer].size();
            int cols = cgradients[layer][0].size();
            matrix_vector_average<<<calculate_grid_2d(cols, rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), block_2d>>>(d_totalCgrad, d_gradC[layer], batchSize, rows, cols);
            matrix_vector_average<<<calculate_grid_2d(cols, rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), block_2d>>>(d_totalBgrad, d_gradB[layer], batchSize, rows, cols);
        }

        // --- Backpropagation for the First Layer (Layer 0) ---
        int cweight_rows = inWidth;
        int cweight_cols = cweights[0][0].size();
        size_t cweight_flat_size = cweight_rows * cweight_cols;
        dim3 gridWeightGradFirst = calculate_grid_1d(cweight_flat_size, CUDA_BLOCK_SIZE_1D);
        int rows = cgradients[0].size();
        int cols = cgradients[0][0].size();
        
        for(int i = 0; i < batchSize; i++) {
            size_t in_flat_size = inHeight * inWidth;
            dim3 grid1D_in = calculate_grid_1d(in_flat_size, CUDA_BLOCK_SIZE_1D);

            // 1. input^n: x -> x^n
            float *d_input_p = nullptr; CUDA_CHECK(cudaMalloc(&d_input_p, sizeof(float) * in_flat_size));
            power<<<grid1D_in, block_1d>>>(d_in[i], d_input_p, order, (int)in_flat_size);
            // 2. Transpose x^n: (H x W) -> (W x H)
            float *d_input_p_T = nullptr; size_t in_p_T_size = inWidth * inHeight;
            dim3 grid2D_in = calculate_grid_2d(inWidth, inHeight, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y);
            CUDA_CHECK(cudaMalloc(&d_input_p_T, sizeof(float) * in_p_T_size));
            transpose<<<grid2D_in, block_2d>>>(d_input_p, d_input_p_T, inHeight, inWidth); CUDA_CHECK(cudaFree(d_input_p));
            // 3. dL/dC_1
            float *d_gradC_i = nullptr; CUDA_CHECK(cudaMalloc(&d_gradC_i, sizeof(float) * cweight_flat_size));
            matxmat2mat<<<calculate_grid_2d(cweight_cols, cweight_rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), block_2d>>>(d_input_p_T, d_incoming[0][i], d_gradC_i, cweight_rows, inHeight, cweight_cols);
            scaleByValue<<<gridWeightGradFirst, block_1d>>>(d_gradC_i, d_gradC_i, alpha, (int)cweight_flat_size);
            CUDA_CHECK(cudaMemcpy(d_totalCgrad + i * cweight_flat_size, d_gradC_i, sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice)); CUDA_CHECK(cudaFree(d_gradC_i));
            // 4. dL/dB_1
            float *d_ones_i = nullptr; CUDA_CHECK(cudaMalloc(&d_ones_i, sizeof(float) * in_flat_size));
            float *d_onesT_i = nullptr; CUDA_CHECK(cudaMalloc(&d_onesT_i, sizeof(float) * in_p_T_size));
            std::vector<float> ones(in_flat_size, 1.0f); CUDA_CHECK(cudaMemcpy(d_ones_i, ones.data(), sizeof(float) * in_flat_size, cudaMemcpyHostToDevice));
            transpose<<<grid2D_in, block_2d>>>(d_ones_i, d_onesT_i, inHeight, inWidth); CUDA_CHECK(cudaFree(d_ones_i));
            float *d_gradB_i = nullptr; CUDA_CHECK(cudaMalloc(&d_gradB_i, sizeof(float) * cweight_flat_size));
            matxmat2mat<<<calculate_grid_2d(cweight_cols, cweight_rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), block_2d>>>(d_onesT_i, d_incoming[0][i], d_gradB_i, cweight_rows, inHeight, cweight_cols);
            scaleByValue<<<gridWeightGradFirst, block_1d>>>(d_gradB_i, d_gradB_i, 1.0f - alpha, (int)cweight_flat_size);
            CUDA_CHECK(cudaMemcpy(d_totalBgrad + i * cweight_flat_size, d_gradB[0], sizeof(float) * cweight_flat_size, cudaMemcpyDeviceToDevice)); CUDA_CHECK(cudaFree(d_gradB_i));
            CUDA_CHECK(cudaFree(d_input_p_T)); CUDA_CHECK(cudaFree(d_onesT_i));
        }
        // Average the gradients
        matrix_vector_average<<<calculate_grid_2d(cols, rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), block_2d>>>(d_totalCgrad, d_gradC[0], batchSize, rows, cols);
        matrix_vector_average<<<calculate_grid_2d(cols, rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), block_2d>>>(d_totalBgrad, d_gradB[0], batchSize, rows, cols);

        // --- Update Weights and Copy Results Back --- (Same as mnn)
        for (int i = 0; i < this->layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            dim3 gridUpdate = calculate_grid_1d(c_size, CUDA_BLOCK_SIZE_1D);
            kernelUpdateWeightsElasticNet<<<gridUpdate, block_1d>>>(d_cweights[i], d_gradC[i], (int)c_size, learningRate, LAMBDA_L1, LAMBDA_L2);
            kernelUpdateWeightsElasticNet<<<gridUpdate, block_1d>>>(d_bweights[i], d_gradB[i], (int)b_size, learningRate, LAMBDA_L1, LAMBDA_L2);
            std::vector<float> c_upd(c_size), b_upd(b_size), cg_upd(c_size), bg_upd(b_size);
            CUDA_CHECK(cudaMemcpy(c_upd.data(), d_cweights[i], sizeof(float) * c_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(b_upd.data(), d_bweights[i], sizeof(float) * b_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(cg_upd.data(), d_gradC[i], sizeof(float) * c_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(bg_upd.data(), d_gradB[i], sizeof(float) * b_size, cudaMemcpyDeviceToHost));
            cweights[i] = reshape(c_upd, cweights[i].size(), cweights[i][0].size());
            bweights[i] = reshape(b_upd, bweights[i].size(), bweights[i][0].size());
            cgradients[i] = reshape(cg_upd, cgradients[i].size(), cgradients[i][0].size());
            bgradients[i] = reshape(bg_upd, bgradients[i].size(), bgradients[i][0].size());
        }

        // --- Cleanup ---
        CUDA_CHECK(cudaFree(d_totalCgrad)); CUDA_CHECK(cudaFree(d_totalBgrad));
        CUDA_CHECK(cudaFree(d_grad_x_CT)); CUDA_CHECK(cudaFree(d_dprev_p)); CUDA_CHECK(cudaFree(d_dprev_act));
        for (int i = 0; i < batchSize; i++) {
            CUDA_CHECK(cudaFree(d_in[i])); CUDA_CHECK(cudaFree(d_exp[i])); CUDA_CHECK(cudaFree(d_out[i])); CUDA_CHECK(cudaFree(d_err[i]));
        }
        for (int i = 0; i < this->layers; i++) {
            CUDA_CHECK(cudaFree(d_cweights[i])); CUDA_CHECK(cudaFree(d_bweights[i]));
            CUDA_CHECK(cudaFree(d_gradC[i])); CUDA_CHECK(cudaFree(d_gradB[i]));
            for(int j = 0; j < batchSize; j++) {
                CUDA_CHECK(cudaFree(d_dotProds[i][j])); CUDA_CHECK(cudaFree(d_activate[i][j]));
                CUDA_CHECK(cudaFree(d_incoming[i][j]));
            }
        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn2d::cuBackprop (batch): ") + e.what());
    }
}

#endif // USE_CUDA