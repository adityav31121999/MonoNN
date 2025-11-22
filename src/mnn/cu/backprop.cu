#ifdef USE_CU
#include "mnn.hpp"
#include "mnn2d.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm> // For std::copy
#include <cmath> // For std::ceil
#include <iostream>

/**
 * @brief Backpropagation for mnn using CUDA.
 * @param expected The expected output vector.
 */
void mnn::cuBackprop(const std::vector<float>& expected) {
    try {
        dim3 block_1d(WORKSIZE_1D);
        dim3 block_2d(WORKSIZE_2D_X, WORKSIZE_2D_Y);
        size_t inputSize = input.size();
        size_t outputSize = output.size();

        float *d_in = nullptr, *d_exp = nullptr, *d_out = nullptr, *d_err = nullptr, *d_ones = nullptr;
        std::vector<float*> d_incoming(this->layers, nullptr);
        std::vector<float*> d_cweights(this->layers, nullptr);
        std::vector<float*> d_bweights(this->layers, nullptr);
        std::vector<float*> d_gradC(this->layers, nullptr);
        std::vector<float*> d_gradB(this->layers, nullptr);
        std::vector<float*> d_dotProds(this->layers, nullptr);
        std::vector<float*> d_activate(this->layers, nullptr);
        std::vector<float*> d_dpow(this->layers - 1, nullptr);
        std::vector<float*> d_dact(this->layers - 1, nullptr);
        float *d_preoutgoing_l = nullptr; // For d_preoutgoing[layer - 1]
        float *d_outgoing_l = nullptr;   // For d_outgoing[layer - 1]

        CUDA_CHECK(cudaMalloc(&d_in, sizeof(float) * inputSize));
        CUDA_CHECK(cudaMalloc(&d_exp, sizeof(float) * outputSize));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * outputSize));
        CUDA_CHECK(cudaMalloc(&d_err, sizeof(float) * outputSize));
        CUDA_CHECK(cudaMemcpy(d_in, input.data(), sizeof(float) * inputSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_exp, expected.data(), sizeof(float) * outputSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out, output.data(), sizeof(float) * outputSize, cudaMemcpyHostToDevice));

        dim3 gridSub = calculate_grid_1d(outputSize, WORKSIZE_1D);
        subtract<<<gridSub, block_1d>>>(d_out, d_exp, d_err, (int)outputSize);
        
        for(int i = 0; i < this->layers; i++) {
            std::vector<float> flat_c = flatten(cweights[i]);
            std::vector<float> flat_b = flatten(bweights[i]);
            size_t cweight_size = flat_c.size();
            size_t bweight_size = flat_b.size();
            size_t act_size = activate[i].size();
            // allocate and copy weights, biases, activations, dot products
            CUDA_CHECK(cudaMalloc(&d_cweights[i], sizeof(float) * cweight_size));
            CUDA_CHECK(cudaMalloc(&d_bweights[i], sizeof(float) * bweight_size));
            CUDA_CHECK(cudaMalloc(&d_gradC[i], sizeof(float) * cweight_size));
            CUDA_CHECK(cudaMalloc(&d_gradB[i], sizeof(float) * bweight_size));
            CUDA_CHECK(cudaMalloc(&d_activate[i], sizeof(float) * act_size));
            CUDA_CHECK(cudaMalloc(&d_dotProds[i], sizeof(float) * act_size));
            CUDA_CHECK(cudaMalloc(&d_incoming[i], sizeof(float) * act_size));
            CUDA_CHECK(cudaMemcpy(d_cweights[i], flat_c.data(), sizeof(float) * cweight_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_bweights[i], flat_b.data(), sizeof(float) * bweight_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_activate[i], activate[i].data(), sizeof(float) * act_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_dotProds[i], dotProds[i].data(), sizeof(float) * act_size, cudaMemcpyHostToDevice));
        }

        for(int i = 0; i < this->layers - 1; i++) {
            size_t act_size = activate[i].size();
            CUDA_CHECK(cudaMalloc(&d_dpow[i], sizeof(float) * act_size));
            CUDA_CHECK(cudaMalloc(&d_dact[i], sizeof(float) * act_size));
        }

        size_t max_outgoing_size = 0;
        if (layers > 1) {
            for (int i = 0; i < layers - 1; i++) {
                max_outgoing_size = std::max(max_outgoing_size, activate[i].size());
            }
        }
        if (max_outgoing_size > 0) {
            CUDA_CHECK(cudaMalloc(&d_preoutgoing_l, sizeof(float) * max_outgoing_size));
            CUDA_CHECK(cudaMalloc(&d_outgoing_l, sizeof(float) * max_outgoing_size));
        }
        CUDA_CHECK(cudaMemcpy(d_incoming[layers - 1], d_err, sizeof(float) * outputSize, cudaMemcpyDeviceToDevice));
        
        std::vector<float> v1(outputSize, 1.0f); // Max size needed for 1D MNN is max layer width
        size_t max_layer_width = 0;
        for (const auto& w : width) { max_layer_width = std::max(max_layer_width, (size_t)w); }
        max_layer_width = std::max(max_layer_width, outputSize);
        v1.resize(max_layer_width, 1.0f);
        CUDA_CHECK(cudaMalloc(&d_ones, sizeof(float) * max_layer_width));
        CUDA_CHECK(cudaMemcpy(d_ones, v1.data(), sizeof(float) * max_layer_width, cudaMemcpyHostToDevice));

        // Backpropagation loop (last layer to second layer)
        for(int layer = layers - 1; layer >= 1; layer--) {
            int prev_size = activate[layer - 1].size();
            int curr_size = activate[layer].size();
            int cweight_rows = prev_size;
            int cweight_cols = curr_size;
            size_t cweight_flat_size = cweight_rows * cweight_cols;

            dim3 gridWeightGrad = calculate_grid_1d(cweight_flat_size, WORKSIZE_1D);
            dim3 gridOutGrad = calculate_grid_1d(prev_size, WORKSIZE_1D);

            // dL/dC_l (Outer Product: d_incoming[L] x d_activate[L-1])
            vecxvec2mat<<<gridWeightGrad, block_1d>>>(
                d_activate[layer - 1], d_incoming[layer], d_gradC[layer], cweight_rows, cweight_cols
            );
            scaleByValue<<<gridWeightGrad, block_1d>>>(
                d_gradC[layer], d_gradC[layer], alpha, (int)cweight_flat_size
            );

            // dL/dB_l (Outer Product: ones x d_incoming[L])
            vecxvec2mat<<<gridWeightGrad, block_1d>>>(
                d_ones, d_incoming[layer], d_gradB[layer], cweight_rows, cweight_cols
            );
            scaleByValue<<<gridWeightGrad, block_1d>>>(
                d_gradB[layer], d_gradB[layer], alpha, (int)cweight_flat_size
            );

            // Outgoing Gradient Calculation (for layer-1)
            float *d_C_T = nullptr;
            CUDA_CHECK(cudaMalloc(&d_C_T, sizeof(float) * cweight_flat_size));
            transpose<<<calculate_grid_2d(cweight_cols, cweight_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                d_cweights[layer],
                d_C_T,
                cweight_rows, // rows
                cweight_cols  // cols
            );

            // incoming gradient x C^T
            vecxmat2vec<<<gridOutGrad, block_1d>>>(
                d_incoming[layer], d_C_T, d_preoutgoing_l, cweight_cols, cweight_rows
            );
            CUDA_CHECK(cudaFree(d_C_T));

            dPower<<<gridOutGrad, block_1d>>>(
                d_activate[layer - 1], d_dpow[layer - 1], order, prev_size
            );
            sigmoidDer<<<gridOutGrad, block_1d>>>(
                d_dotProds[layer - 1], d_dact[layer - 1], prev_size
            );
            hadamard2<<<gridOutGrad, block_1d>>>(
                d_preoutgoing_l, d_dpow[layer - 1], d_dact[layer - 1], d_incoming[layer - 1], 1, prev_size
            );
        }

        // Backpropagation for the first layer (input layer)
        int prev_size = input.size();
        int curr_size = activate[0].size();
        int cweight_rows = prev_size;
        int cweight_cols = curr_size;
        size_t cweight_flat_size = cweight_rows * cweight_cols;
        dim3 gridWeightGrad = calculate_grid_1d(cweight_flat_size, WORKSIZE_1D);

        // 1. dL/dC_1 (Outer Product: d_incoming[0] x d_in)
        vecxvec2mat<<<gridWeightGrad, block_1d>>>(
            d_in, d_incoming[0], d_gradC[0], cweight_rows, cweight_cols
        );
        scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradC[0], d_gradC[0], alpha, (int)cweight_flat_size);

        vecxvec2mat<<<gridWeightGrad, block_1d>>>(
            d_ones, d_incoming[0], d_gradB[0], cweight_rows, cweight_cols
        );
        scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradB[0], d_gradB[0], alpha, (int)cweight_flat_size);

        // --- Update Weights and Copy Results Back ---
        for (int i = 0; i < this->layers; ++i) {
            std::vector<float> flat_c = flatten(cweights[i]);
            std::vector<float> flat_b = flatten(bweights[i]);
            size_t c_size = flat_c.size();
            size_t b_size = flat_b.size();
            dim3 gridUpdate = calculate_grid_1d(c_size, WORKSIZE_1D);

            kernelUpdateWeightsElasticNet<<<gridUpdate, block_1d>>>(
                d_cweights[i], d_gradC[i], (int)c_size, learningRate, LAMBDA_L1, LAMBDA_L2
            );
            kernelUpdateWeightsElasticNet<<<gridUpdate, block_1d>>>(
                d_bweights[i], d_gradB[i], (int)b_size, learningRate, LAMBDA_L1, LAMBDA_L2
            );

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

        // free memory
        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_exp));
        CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_err));
        CUDA_CHECK(cudaFree(d_ones)); CUDA_CHECK(cudaFree(d_preoutgoing_l));
        CUDA_CHECK(cudaFree(d_outgoing_l));
        for (int i = 0; i < this->layers; i++) {
            CUDA_CHECK(cudaFree(d_cweights[i])); CUDA_CHECK(cudaFree(d_bweights[i]));
            CUDA_CHECK(cudaFree(d_gradC[i])); CUDA_CHECK(cudaFree(d_gradB[i]));
            CUDA_CHECK(cudaFree(d_dotProds[i])); CUDA_CHECK(cudaFree(d_activate[i]));
            CUDA_CHECK(cudaFree(d_incoming[i]));
            if (i < this->layers - 1) {
                CUDA_CHECK(cudaFree(d_dpow[i])); CUDA_CHECK(cudaFree(d_dact[i]));
            }
        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn::cuBackprop: ") + e.what());
    }
}


/**
 * @brief Backpropagation for mnn2d using CUDA.
 * @param expected The expected output vector.
 */
void mnn2d::cuBackprop(const std::vector<float>& expected)
{
    float *d_in = nullptr, *d_exp = nullptr, *d_out = nullptr, *d_err = nullptr;
    float *d_grad_x_CT = nullptr;

    std::vector<float*> d_incoming(this->layers, nullptr);
    std::vector<float*> d_cweights(this->layers, nullptr);
    std::vector<float*> d_bweights(this->layers, nullptr);
    std::vector<float*> d_gradC(this->layers, nullptr);
    std::vector<float*> d_gradB(this->layers, nullptr);
    std::vector<float*> d_dotProds(this->layers, nullptr);
    std::vector<float*> d_activate(this->layers, nullptr);
    std::vector<float*> d_outgoing(this->layers - 1, nullptr);
    std::vector<float*> d_dpow(this->layers - 1, nullptr);
    std::vector<float*> d_dact(this->layers - 1, nullptr);

    float *d_C_T = nullptr, *d_prev_p = nullptr, *d_prev_p_T = nullptr;
    float *d_input_p = nullptr, *d_input_p_T = nullptr;
    float *d_onesT = nullptr;   // *d_ones = nullptr;
    
    size_t max_act_size = 0;
    for (int i = 0; i < layers; ++i) {
        max_act_size = std::max(max_act_size, activate[i].size() * activate[i][0].size());
    }
    max_act_size = std::max(max_act_size, (size_t)inHeight * inWidth);
    
    try {
        dim3 block_1d(WORKSIZE_1D);
        dim3 block_2d(WORKSIZE_2D_X, WORKSIZE_2D_Y);

        // --- Allocate and Copy Initial Data ---
        CUDA_CHECK(cudaMalloc(&d_in, sizeof(float) * inHeight * inWidth));
        CUDA_CHECK(cudaMalloc(&d_exp, sizeof(float) * inHeight * outWidth));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * inHeight * outWidth));
        CUDA_CHECK(cudaMalloc(&d_err, sizeof(float) * inHeight * outWidth));
        CUDA_CHECK(cudaMemcpy(d_in, flatten(input).data(), sizeof(float) * inHeight * inWidth, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_exp, expected.data(), sizeof(float) * inHeight * outWidth, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out, output.data(), sizeof(float) * inHeight * outWidth, cudaMemcpyHostToDevice));
        // --- Allocate Layer-Specific Buffers ---
        for(int i = 0; i < this->layers; i++) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            size_t dot_size = dotProds[i].size() * dotProds[i][0].size();
            size_t act_size = activate[i].size() * activate[i][0].size();

            CUDA_CHECK(cudaMalloc(&d_cweights[i], sizeof(float) * c_size));
            CUDA_CHECK(cudaMalloc(&d_bweights[i], sizeof(float) * b_size));
            CUDA_CHECK(cudaMemcpy(d_cweights[i], flatten(cweights[i]).data(), sizeof(float) * c_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_bweights[i], flatten(bweights[i]).data(), sizeof(float) * b_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_gradC[i], sizeof(float) * c_size));
            CUDA_CHECK(cudaMalloc(&d_gradB[i], sizeof(float) * b_size));
            CUDA_CHECK(cudaMalloc(&d_dotProds[i], sizeof(float) * dot_size));
            CUDA_CHECK(cudaMalloc(&d_activate[i], sizeof(float) * act_size));
            CUDA_CHECK(cudaMemcpy(d_dotProds[i], flatten(dotProds[i]).data(), sizeof(float) * dot_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_activate[i], flatten(activate[i]).data(), sizeof(float) * act_size, cudaMemcpyHostToDevice));

            if (i < this->layers - 1) {
                // This is for the gradient coming *into* layer i from i+1, so its size is act_size of layer i
                size_t incoming_size = activate[i].size() * activate[i][0].size();
                CUDA_CHECK(cudaMalloc(&d_incoming[i], sizeof(float) * incoming_size));
                CUDA_CHECK(cudaMalloc(&d_outgoing[i], sizeof(float) * act_size));
                CUDA_CHECK(cudaMalloc(&d_dpow[i], sizeof(float) * act_size));
                CUDA_CHECK(cudaMalloc(&d_dact[i], sizeof(float) * act_size));
            }
        }
        
        // backpropagation starts
        // --- Initial Error Calculation (Output Layer) ---
        dim3 gridSub = calculate_grid_1d(inHeight * outWidth, WORKSIZE_1D);
        subtract<<<gridSub, block_1d>>>(d_out, d_exp, d_err, (int)inHeight * outWidth);
        
        // Distribute error for mean pooling
        std::vector<float> out_err_host(inHeight * outWidth);
        CUDA_CHECK(cudaMemcpy(out_err_host.data(), d_err, sizeof(float) * inHeight * outWidth, cudaMemcpyDeviceToHost));
        std::vector<float> last_layer_incoming_flat(activate[layers-1].size() * activate[layers-1][0].size());
        for(size_t r = 0; r < activate[layers-1].size(); ++r) {
            for(size_t c = 0; c < activate[layers-1][0].size(); ++c) {
                last_layer_incoming_flat[r * activate[layers-1][0].size() + c] = out_err_host[c];
            }
        }
        CUDA_CHECK(cudaMalloc(&d_incoming[layers - 1], sizeof(float) * last_layer_incoming_flat.size()));
        CUDA_CHECK(cudaMemcpy(d_incoming[layers - 1], last_layer_incoming_flat.data(), sizeof(float) * last_layer_incoming_flat.size(), cudaMemcpyHostToDevice));

        // --- Allocate Temporary Buffers for Loop ---
        CUDA_CHECK(cudaMalloc(&d_grad_x_CT, sizeof(float) * max_act_size));
        // CUDA_CHECK(cudaMalloc(&d_ones, sizeof(float) * max_act_size));
        // std::vector<float> h_ones(max_act_size, 1.0f);
        // CUDA_CHECK(cudaMemcpy(d_ones, h_ones.data(), sizeof(float) * max_act_size, cudaMemcpyHostToDevice));

        // --- Backpropagation Loop (Last Layer to Second Layer) ---
        for(int layer = layers - 1; layer >= 1; layer--) {
            int prev_rows = activate[layer-1].size();
            int prev_cols = activate[layer-1][0].size();
            int curr_rows = activate[layer].size();
            int curr_cols = activate[layer][0].size();
            int cweight_rows = cweights[layer].size();
            int cweight_cols = cweights[layer][0].size();
            size_t cweight_flat_size = curr_rows * curr_cols;
            size_t prev_act_flat_size = prev_rows * prev_cols;

            dim3 grid_prev_act_flat = calculate_grid_1d(prev_act_flat_size, WORKSIZE_1D);
            dim3 grid_cweight_flat = calculate_grid_1d(cweight_flat_size, WORKSIZE_1D);
            dim3 grid_transpose_c = calculate_grid_2d(cweight_cols, cweight_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y);
            dim3 grid_matmul_grad_ct = calculate_grid_2d(prev_cols, curr_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y);
            dim3 grid_matmul_gradc = calculate_grid_2d(cweight_cols, prev_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y);

            // Transpose C
            CUDA_CHECK(cudaMalloc(&d_C_T, sizeof(float) * cweight_flat_size));
            transpose<<<grid_transpose_c, block_2d>>>(
                d_cweights[layer], d_C_T, cweight_rows, cweight_cols
            );

            // Calculate dL/dz_l * C^T
            matxmat2mat<<<grid_matmul_grad_ct, block_2d>>>(
                d_incoming[layer], d_C_T, d_grad_x_CT, curr_rows, cweight_cols, cweight_rows
            );
            CUDA_CHECK(cudaFree(d_C_T));

            // d(prev_p)
            dPower<<<grid_prev_act_flat, block_1d>>>(
                d_activate[layer-1], d_dpow[layer-1], order, prev_act_flat_size
            );

            // d(prev_act) (Softmax derivative)
            size_t prev_dot_size = dotProds[layer-1].size() * dotProds[layer-1][0].size();
            if (prev_dot_size > WORKSIZE_1D) {
                size_t num_work_groups = (prev_dot_size + WORKSIZE_1D - 1) / WORKSIZE_1D;
                size_t partial_results_buffer_size = num_work_groups * 2;
                float *d_partial_results = nullptr;
                CUDA_CHECK(cudaMalloc(&d_partial_results, sizeof(float) * partial_results_buffer_size));
                softmax_reduce<<<num_work_groups, block_1d, 2 * WORKSIZE_1D * sizeof(float)>>>(d_dotProds[layer-1], d_partial_results, (int)prev_dot_size, SOFTMAX_TEMP);

                std::vector<float> h_partial_results(partial_results_buffer_size);
                CUDA_CHECK(cudaMemcpy(h_partial_results.data(), d_partial_results, sizeof(float) * partial_results_buffer_size, cudaMemcpyDeviceToHost));
                float global_max = -(FLT_MAX);
                float global_sum = 0.0f;
                for (size_t k = 0; k < num_work_groups; ++k) { 
                    global_sum += h_partial_results[2 * k];
                    global_max = fmaxf(global_max, h_partial_results[2 * k + 1]);
                }
                softmaxDer_normalize<<<calculate_grid_1d(prev_dot_size, WORKSIZE_1D), block_1d>>>(
                    d_dotProds[layer-1], d_dact[layer-1], (int)prev_dot_size, SOFTMAX_TEMP, global_max, global_sum
                );
                CUDA_CHECK(cudaFree(d_partial_results));
            }
            else {
                softmaxDer<<<calculate_grid_1d(prev_dot_size, WORKSIZE_1D), block_1d>>>(
                    d_dotProds[layer-1], d_dact[layer-1], SOFTMAX_TEMP, (int)prev_dot_size
                );
            }

            // outgoing gradient
            hadamard2<<<grid_prev_act_flat, block_1d>>>(
                d_grad_x_CT, d_dpow[layer-1], d_dact[layer-1], d_outgoing[layer-1], prev_rows, prev_cols
            );

            // dL/dC_layer
            CUDA_CHECK(cudaMalloc(&d_prev_p, sizeof(float) * prev_act_flat_size));
            power<<<grid_prev_act_flat, block_1d>>>(
                d_activate[layer-1], d_prev_p, order, prev_act_flat_size
            );

            // transpose d_prev_p
            CUDA_CHECK(cudaMalloc(&d_prev_p_T, sizeof(float) * prev_act_flat_size));
            transpose<<<calculate_grid_2d(prev_cols, prev_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                d_prev_p, d_prev_p_T, prev_rows, prev_cols
            );
            // dL/dC_layer
            matxmat2mat<<<grid_matmul_gradc, block_2d>>>(
                d_prev_p_T, d_incoming[layer], d_gradC[layer], prev_cols, prev_rows, cweight_cols
            );
            scaleByValue<<<grid_cweight_flat, block_1d>>>(
                d_gradC[layer], d_gradC[layer], alpha, (int)cweight_flat_size
            );
            CUDA_CHECK(cudaFree(d_prev_p));
            CUDA_CHECK(cudaFree(d_prev_p_T));

            // transpose ones
            CUDA_CHECK(cudaMalloc(&d_onesT, sizeof(float) * prev_act_flat_size));
            std::vector<float> h_ones(prev_act_flat_size, 1.0f); // Host vector of ones
            CUDA_CHECK(cudaMemcpy(d_onesT, h_ones.data(), sizeof(float) * prev_act_flat_size, cudaMemcpyHostToDevice));
            // transpose<<<calculate_grid_2d(prev_cols, prev_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
            //     d_ones, d_onesT, prev_rows, prev_cols
            // );
            // dL/dB_layer
            matxmat2mat<<<grid_matmul_gradc, block_2d>>>(
                d_onesT, d_incoming[layer], d_gradB[layer], prev_cols, prev_rows, cweight_cols
            );
            scaleByValue<<<grid_cweight_flat, block_1d>>>(
                d_gradB[layer], d_gradB[layer], (1.0f - alpha), (int)cweight_flat_size
            );
            CUDA_CHECK(cudaFree(d_onesT));

            // Copy d_outgoing[layer-1] to d_incoming[layer-1]
            CUDA_CHECK(cudaMemcpy(d_incoming[layer-1], d_outgoing[layer-1], sizeof(float) * prev_act_flat_size, cudaMemcpyDeviceToDevice));
        }

        // --- Backpropagation for the First Layer (Layer 0) ---
        int first_layer_out_rows = activate[0].size();
        int first_layer_out_cols = activate[0][0].size();
        size_t first_layer_cweight_flat_size = inWidth * first_layer_out_cols;
        dim3 grid_first_layer_cweight_flat = calculate_grid_1d(first_layer_cweight_flat_size, WORKSIZE_1D);
        dim3 grid_input_flat = calculate_grid_1d(inHeight * inWidth, WORKSIZE_1D);
        dim3 grid_transpose_input = calculate_grid_2d(inWidth, inHeight, WORKSIZE_2D_X, WORKSIZE_2D_Y);
        dim3 grid_matmul_gradc_first = calculate_grid_2d(inWidth, first_layer_out_cols, WORKSIZE_2D_X, WORKSIZE_2D_Y);

        // Calculate d_input_p
        CUDA_CHECK(cudaMalloc(&d_input_p, sizeof(float) * inHeight * inWidth));
        power<<<grid_input_flat, block_1d>>>(
            d_in, d_input_p, order, inHeight * inWidth
        );

        // Transpose d_input_p
        CUDA_CHECK(cudaMalloc(&d_input_p_T, sizeof(float) * inHeight * inWidth));
        transpose<<<grid_transpose_input, block_2d>>>(
            d_input_p, d_input_p_T, inHeight, inWidth
        );

        // Calculate dL/dC_0
        matxmat2mat<<<grid_matmul_gradc_first, block_2d>>>(
            d_input_p_T, d_incoming[0], d_gradC[0], inWidth, inHeight, first_layer_out_cols
        );
        scaleByValue<<<grid_first_layer_cweight_flat, block_1d>>>(
            d_gradC[0], d_gradC[0], alpha, (int)first_layer_cweight_flat_size
        );
        CUDA_CHECK(cudaFree(d_input_p));
        CUDA_CHECK(cudaFree(d_input_p_T));

        // Calculate dL/dB_0
        std::vector<float> h_ones(inHeight * inWidth, 1.0f);
        CUDA_CHECK(cudaMalloc(&d_onesT, sizeof(float) * inHeight * inWidth));
        CUDA_CHECK(cudaMemcpy(d_onesT, h_ones.data(), sizeof(float) * inHeight * inWidth, cudaMemcpyHostToDevice));
        // transpose<<<grid_transpose_input, block_2d>>>(
        //     d_ones, d_onesT, inHeight, inWidth
        // );
        matxmat2mat<<<grid_matmul_gradc_first, block_2d>>>(
            d_onesT, d_incoming[0], d_gradB[0], inWidth, inHeight, first_layer_out_cols
        );
        scaleByValue<<<grid_first_layer_cweight_flat, block_1d>>>(
            d_gradB[0], d_gradB[0], (1.0f - alpha), (int)first_layer_cweight_flat_size
        );
        CUDA_CHECK(cudaFree(d_onesT));

        // --- Update Weights and Copy Results Back ---
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
            CUDA_CHECK(cudaMemcpy(c_upd.data(), d_cweights[i], sizeof(float) * c_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(b_upd.data(), d_bweights[i], sizeof(float) * b_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(cg_upd.data(), d_gradC[i], sizeof(float) * c_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(bg_upd.data(), d_gradB[i], sizeof(float) * b_size, cudaMemcpyDeviceToHost));

            cweights[i] = reshape(c_upd, cweights[i].size(), cweights[i][0].size());
            bweights[i] = reshape(b_upd, bweights[i].size(), bweights[i][0].size());
            cgradients[i] = reshape(cg_upd, cgradients[i].size(), cgradients[i][0].size());
            bgradients[i] = reshape(bg_upd, bgradients[i].size(), bgradients[i][0].size());
        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn2d::cuBackprop: ") + e.what());
    }
    
    // --- FINAL MEMORY CLEANUP BLOCK ---
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_exp));
    CUDA_CHECK(cudaFree(d_err));
    CUDA_CHECK(cudaFree(d_grad_x_CT));
    // CUDA_CHECK(cudaFree(d_ones));

    // Cleanup layer-specific vectors
    for (int i = 0; i < this->layers; i++) {
        CUDA_CHECK(cudaFree(d_cweights[i])); CUDA_CHECK(cudaFree(d_bweights[i]));
        CUDA_CHECK(cudaFree(d_gradC[i])); CUDA_CHECK(cudaFree(d_gradB[i]));
        CUDA_CHECK(cudaFree(d_incoming[i]));
        CUDA_CHECK(cudaFree(d_dotProds[i])); CUDA_CHECK(cudaFree(d_activate[i]));
        // These vectors only have layers-1 elements
        if (i < this->layers - 1) {
            CUDA_CHECK(cudaFree(d_outgoing[i]));
            CUDA_CHECK(cudaFree(d_dpow[i]));
            CUDA_CHECK(cudaFree(d_dact[i]));
        }
    }
}

#endif