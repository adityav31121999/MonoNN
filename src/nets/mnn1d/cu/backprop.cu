#ifdef USE_CU
#include "mnn1d.hpp"

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstdlib> // For rand()

/**
 * @brief Backpropagation for mnn1d using CUDA.
 * @param expected The expected output vector.
 */
void mnn1d::cuBackprop(const std::vector<float>& expected) {
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
        float *d_preoutgoing_l = nullptr;
        float *d_outgoing_l = nullptr;

        CU_CHECK(cudaMalloc(&d_in, sizeof(float) * inputSize));
        CU_CHECK(cudaMalloc(&d_exp, sizeof(float) * outputSize));
        CU_CHECK(cudaMalloc(&d_out, sizeof(float) * outputSize));
        CU_CHECK(cudaMalloc(&d_err, sizeof(float) * outputSize));
        CU_CHECK(cudaMemcpy(d_in, input.data(), sizeof(float) * inputSize, cudaMemcpyHostToDevice));
        CU_CHECK(cudaMemcpy(d_exp, expected.data(), sizeof(float) * outputSize, cudaMemcpyHostToDevice));
        CU_CHECK(cudaMemcpy(d_out, activate[layers - 1].data(), sizeof(float) * outputSize, cudaMemcpyHostToDevice));

        dim3 gridSub = calculate_grid_1d(outputSize, WORKSIZE_1D);
        subtract<<<gridSub, block_1d>>>(d_out, d_exp, d_err, (int)outputSize);
        
        for(int i = 0; i < this->layers; i++) {
            std::vector<float> flat_c = flatten(cweights[i]);
            std::vector<float> flat_b = flatten(bweights[i]);
            size_t cweight_size = flat_c.size();
            size_t bweight_size = flat_b.size();
            size_t act_size = activate[i].size();
            // allocate and copy weights, biases, activations, dot products
            CU_CHECK(cudaMalloc(&d_cweights[i], sizeof(float) * cweight_size));
            CU_CHECK(cudaMalloc(&d_bweights[i], sizeof(float) * bweight_size));
            CU_CHECK(cudaMalloc(&d_gradC[i], sizeof(float) * cweight_size));
            CU_CHECK(cudaMalloc(&d_gradB[i], sizeof(float) * bweight_size));
            CU_CHECK(cudaMalloc(&d_activate[i], sizeof(float) * act_size));
            CU_CHECK(cudaMalloc(&d_dotProds[i], sizeof(float) * act_size));
            CU_CHECK(cudaMalloc(&d_incoming[i], sizeof(float) * act_size));
            CU_CHECK(cudaMemcpy(d_cweights[i], flat_c.data(), sizeof(float) * cweight_size, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(d_bweights[i], flat_b.data(), sizeof(float) * bweight_size, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(d_activate[i], activate[i].data(), sizeof(float) * act_size, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(d_dotProds[i], dotProds[i].data(), sizeof(float) * act_size, cudaMemcpyHostToDevice));
        }

        for(int i = 0; i < this->layers - 1; i++) {
            size_t act_size = activate[i].size();
            CU_CHECK(cudaMalloc(&d_dpow[i], sizeof(float) * act_size));
            CU_CHECK(cudaMalloc(&d_dact[i], sizeof(float) * act_size));
        }

        size_t max_outgoing_size = 0;
        if (layers > 1) {
            for (int i = 0; i < layers - 1; i++) {
                max_outgoing_size = std::max(max_outgoing_size, activate[i].size());
            }
        }
        if (max_outgoing_size > 0) {
            CU_CHECK(cudaMalloc(&d_preoutgoing_l, sizeof(float) * max_outgoing_size));
            CU_CHECK(cudaMalloc(&d_outgoing_l, sizeof(float) * max_outgoing_size));
        }
        CU_CHECK(cudaMemcpy(d_incoming[layers - 1], d_err, sizeof(float) * outputSize, cudaMemcpyDeviceToDevice));
        
        std::vector<float> v1(outputSize, 1.0f); // Max size needed for 1D MNN is max layer width
        size_t max_layer_width = 0;
        for (const auto& w : width) { max_layer_width = std::max(max_layer_width, (size_t)w); }
        max_layer_width = std::max(max_layer_width, outputSize);
        v1.resize(max_layer_width, 1.0f);
        CU_CHECK(cudaMalloc(&d_ones, sizeof(float) * max_layer_width));
        CU_CHECK(cudaMemcpy(d_ones, v1.data(), sizeof(float) * max_layer_width, cudaMemcpyHostToDevice));

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

            // dL/dB_l (Outer Product: ones x d_incoming[L])
            vecxvec2mat<<<gridWeightGrad, block_1d>>>(
                d_ones, d_incoming[layer], d_gradB[layer], cweight_rows, cweight_cols
            );

            scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradB[layer], d_gradB[layer], 1.0f - ALPHA, (int)cweight_flat_size);
            scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradC[layer], d_gradC[layer], ALPHA, (int)cweight_flat_size);

            // Outgoing Gradient Calculation (for layer-1)
            float *d_C_T = nullptr;
            CU_CHECK(cudaMalloc(&d_C_T, sizeof(float) * cweight_flat_size));
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
            CU_CHECK(cudaFree(d_C_T));

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
        scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradC[0], d_gradC[0], ALPHA, (int)cweight_flat_size);

        vecxvec2mat<<<gridWeightGrad, block_1d>>>(
            d_ones, d_incoming[0], d_gradB[0], cweight_rows, cweight_cols
        );
        scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradB[0], d_gradB[0], 1.0f - ALPHA, (int)cweight_flat_size);

        // --- Update Weights and Copy Results Back ---
        for (int i = 0; i < this->layers; ++i) {
            std::vector<float> flat_c = flatten(cweights[i]);
            std::vector<float> flat_b = flatten(bweights[i]);
            size_t c_size = flat_c.size();
            size_t b_size = flat_b.size();
            dim3 gridUpdate = calculate_grid_1d(c_size, WORKSIZE_1D);

            switch (weightUpdateType) {
                case 0:
                    kernelUpdateWeights<<<gridUpdate, block_1d>>>(d_cweights[i], d_gradC[i], learningRate, (int)c_size);
                    kernelUpdateWeights<<<calculate_grid_1d(b_size, WORKSIZE_1D), block_1d>>>(d_bweights[i], d_gradB[i], learningRate, (int)b_size);
                    break;
                case 1:
                    kernelUpdateWeightsWithL1<<<gridUpdate, block_1d>>>(d_cweights[i], d_gradC[i], (int)c_size, learningRate, LAMBDA_L1);
                    kernelUpdateWeightsWithL1<<<calculate_grid_1d(b_size, WORKSIZE_1D), block_1d>>>(d_bweights[i], d_gradB[i], (int)b_size, learningRate, LAMBDA_L1);
                    break;
                case 2:
                    kernelUpdateWeightsWithL2<<<gridUpdate, block_1d>>>(d_cweights[i], d_gradC[i], (int)c_size, learningRate, LAMBDA_L2);
                    kernelUpdateWeightsWithL2<<<calculate_grid_1d(b_size, WORKSIZE_1D), block_1d>>>(d_bweights[i], d_gradB[i], (int)b_size, learningRate, LAMBDA_L2);
                    break;
                case 3:
                    kernelUpdateWeightsElasticNet<<<gridUpdate, block_1d>>>(d_cweights[i], d_gradC[i], (int)c_size, learningRate, LAMBDA_L1, LAMBDA_L2);
                    kernelUpdateWeightsElasticNet<<<calculate_grid_1d(b_size, WORKSIZE_1D), block_1d>>>(d_bweights[i], d_gradB[i], (int)b_size, learningRate, LAMBDA_L1, LAMBDA_L2);
                    break;
                case 4:
                    kernelUpdateWeightsWithWeightDecay<<<gridUpdate, block_1d>>>(d_cweights[i], d_gradC[i], (int)c_size, learningRate, WEIGHT_DECAY);
                    kernelUpdateWeightsWithWeightDecay<<<calculate_grid_1d(b_size, WORKSIZE_1D), block_1d>>>(d_bweights[i], d_gradB[i], (int)b_size, learningRate, WEIGHT_DECAY);
                    break;
                case 5:
                    kernelUpdateWeightsDropout<<<gridUpdate, block_1d>>>(d_cweights[i], d_gradC[i], (int)c_size, learningRate, DROPOUT_RATE, (unsigned int)rand());
                    kernelUpdateWeightsDropout<<<calculate_grid_1d(b_size, WORKSIZE_1D), block_1d>>>(d_bweights[i], d_gradB[i], (int)b_size, learningRate, DROPOUT_RATE, (unsigned int)rand());
                    break;
            }

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

        // free memory
        CU_CHECK(cudaFree(d_in)); CU_CHECK(cudaFree(d_exp));
        CU_CHECK(cudaFree(d_out)); CU_CHECK(cudaFree(d_err));
        CU_CHECK(cudaFree(d_ones)); CU_CHECK(cudaFree(d_preoutgoing_l));
        CU_CHECK(cudaFree(d_outgoing_l));
        for (int i = 0; i < this->layers; i++) {
            CU_CHECK(cudaFree(d_cweights[i])); CU_CHECK(cudaFree(d_bweights[i]));
            CU_CHECK(cudaFree(d_gradC[i])); CU_CHECK(cudaFree(d_gradB[i]));
            CU_CHECK(cudaFree(d_dotProds[i])); CU_CHECK(cudaFree(d_activate[i]));
            CU_CHECK(cudaFree(d_incoming[i]));
            if (i < this->layers - 1) {
                CU_CHECK(cudaFree(d_dpow[i])); CU_CHECK(cudaFree(d_dact[i]));
            }
        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn1d::cuBackprop: ") + e.what());
    }
}

#endif