#ifdef USE_CU
#include "mnn1d.hpp"

#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm> // For std::max
#include <cmath>     // For std::ceil
#include <limits>    // For std::numeric_limits
#include <cstdlib>   // For rand()

/**
 * @brief trains the mnn1d network on a single input-target pair for 1 cycle using CUDA.
 * @param input The input vector.
 * @param target The target output vector.
 * @param useBuffer 0 for stand alone functions or 1 for all-buffers-in-single function 
 */
void mnn1d::cuTrain1c(const std::vector<float>& input, const std::vector<float>& target, bool useBuffer) {
    if (useBuffer == 0) {
        // 1. Forward propagation
        cuForprop(input);

        if(maxIndex(output) == maxIndex(target)) {
            // std::cout << "Correct output predicted with loss " << crossEntropy(output, target) << "." << std::endl;
        }
        else {
            // check for error and break if acceptable
            currloss = crossEntropy(output, target);
            // std::cout << "Current CE Loss: " << currloss << std::endl;

            // 2. Backward propagation
            cuBackprop(target);
            prevloss = currloss;
        }
    }
    else {
        // 1. Forward propagation
        dim3 local_1d(WORKSIZE_1D);
        dim3 local_2d(WORKSIZE_2D_X, WORKSIZE_2D_Y);

        // --- Buffer Allocation ---
        float *d_in = nullptr, *d_exp = nullptr, *d_out = nullptr, *d_err = nullptr, *d_ones = nullptr;
        std::vector<float*> d_cweights(layers), d_bweights(layers), d_gradC(layers), d_gradB(layers);
        std::vector<float*> d_dotProds(layers), d_activate(layers), d_incoming(layers);
        float *d_preoutgoing_l = nullptr, *d_dpow_l = nullptr, *d_dact_l = nullptr;
        float *d_C_T = nullptr;

        // Allocate input/output/target buffers
        CU_CHECK(cudaMalloc(&d_in, input.size() * sizeof(float)));
        CU_CHECK(cudaMemcpy(d_in, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
        CU_CHECK(cudaMalloc(&d_exp, target.size() * sizeof(float)));
        CU_CHECK(cudaMemcpy(d_exp, target.data(), target.size() * sizeof(float), cudaMemcpyHostToDevice));
        CU_CHECK(cudaMalloc(&d_out, output.size() * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_err, output.size() * sizeof(float)));

        size_t max_layer_width = 0;
        for (int w : width) max_layer_width = std::max(max_layer_width, (size_t)w);
        max_layer_width = std::max(max_layer_width, output.size());
        max_layer_width = std::max(max_layer_width, input.size());
        std::vector<float> v1(max_layer_width, 1.0f);
        CU_CHECK(cudaMalloc(&d_ones, sizeof(float) * max_layer_width));
        CU_CHECK(cudaMemcpy(d_ones, v1.data(), sizeof(float) * max_layer_width, cudaMemcpyHostToDevice));

        // Allocate layer-specific buffers
        for (int i = 0; i < layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            size_t act_size = activate[i].size();
            CU_CHECK(cudaMalloc(&d_cweights[i], c_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_bweights[i], b_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_gradC[i], c_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_gradB[i], b_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_dotProds[i], act_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_activate[i], act_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_incoming[i], act_size * sizeof(float)));
        }

        if (layers > 1) {
            CU_CHECK(cudaMalloc(&d_preoutgoing_l, sizeof(float) * max_layer_width));
            CU_CHECK(cudaMalloc(&d_dpow_l, sizeof(float) * max_layer_width));
            CU_CHECK(cudaMalloc(&d_dact_l, sizeof(float) * max_layer_width));
            size_t max_cweight_size = 0;
            for(int i = 0; i < layers; i++) max_cweight_size = std::max(max_cweight_size, cweights[i].size() * cweights[i][0].size());
            CU_CHECK(cudaMalloc(&d_C_T, sizeof(float) * max_cweight_size));
        }

        // --- Training Loop ---
        // Copy weights H2D for current iteration
        for (int i = 0; i < layers; ++i) {
            std::vector<float> flat_c = flatten(cweights[i]);
            std::vector<float> flat_b = flatten(bweights[i]);
            CU_CHECK(cudaMemcpy(d_cweights[i], flat_c.data(), flat_c.size() * sizeof(float), cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(d_bweights[i], flat_b.data(), flat_b.size() * sizeof(float), cudaMemcpyHostToDevice));
        }

        // --- Forward Propagation ---
        float* d_current_act = d_in;

        // First layer
        int current_in_size = input.size();
        int current_out_size = width[0];
        dim3 globalForward = calculate_grid_1d(current_out_size, WORKSIZE_1D);
        kernelLayerForward2<<<globalForward, local_1d>>>(d_current_act, d_dotProds[0], d_cweights[0], d_bweights[0], current_in_size, current_out_size, order);
        CU_CHECK(cudaGetLastError());

        dim3 globalSig = calculate_grid_1d(current_out_size, WORKSIZE_1D);
        sigmoid<<<globalSig, local_1d>>>(d_dotProds[0], d_activate[0], current_out_size);
        CU_CHECK(cudaGetLastError());

        // Subsequent layers
        for (int i = 1; i < layers; ++i) {
            d_current_act = d_activate[i - 1];
            current_in_size = width[i - 1];
            current_out_size = width[i];
            globalForward = calculate_grid_1d(current_out_size, WORKSIZE_1D);
            kernelLayerForward2<<<globalForward, local_1d>>>(d_current_act, d_dotProds[i], d_cweights[i], d_bweights[i], current_in_size, current_out_size, order);
            CU_CHECK(cudaGetLastError());

            globalSig = calculate_grid_1d(current_out_size, WORKSIZE_1D);
            sigmoid<<<globalSig, local_1d>>>(d_dotProds[i], d_activate[i], current_out_size);
            CU_CHECK(cudaGetLastError());
        }
        CU_CHECK(cudaDeviceSynchronize());

        // Copy output D2H to check for correctness and loss
        CU_CHECK(cudaMemcpy(output.data(), d_activate[layers - 1], output.size() * sizeof(float), cudaMemcpyDeviceToHost));
        output = softmax(output);

        if(maxIndex(output) == maxIndex(target)) {
            float loss = crossEntropy(output, target);
            if (loss < 0) loss = 0;
            // std::cout << "Correct output predicted with loss " << loss << "." << std::endl;
        }
        else {
            currloss = crossEntropy(output, target);
            // std::cout << "Current CE Loss: " << currloss << std::endl;

            // 2. Backward propagation
            zeroGradients();

            CU_CHECK(cudaMemcpy(d_out, d_activate[layers-1], output.size() * sizeof(float), cudaMemcpyDeviceToDevice));
            // 2. Backward propagation
            // Calculate initial error (output - expected)
            dim3 globalSub = calculate_grid_1d(output.size(), WORKSIZE_1D);
            subtract<<<globalSub, local_1d>>>(d_out, d_exp, d_err, (int)output.size());
            CU_CHECK(cudaGetLastError());
            CU_CHECK(cudaMemcpy(d_incoming[layers - 1], d_err, sizeof(float) * output.size(), cudaMemcpyDeviceToDevice));

            // Backpropagation loop (last layer to second layer)
            for (int layer = layers - 1; layer >= 1; --layer) {
                int prev_size = activate[layer - 1].size();
                int curr_size = activate[layer].size();
                int cweight_rows = prev_size;
                int cweight_cols = curr_size;
                size_t cweight_flat_size = cweight_rows * cweight_cols;

                dim3 globalWeightGrad = calculate_grid_1d(cweight_flat_size, WORKSIZE_1D);
                dim3 globalOutGrad = calculate_grid_1d(prev_size, WORKSIZE_1D);

                // dL/dC_l (Outer Product: d_activate[L-1] x d_incoming[L])
                vecxvec2mat<<<globalWeightGrad, local_1d>>>(d_activate[layer - 1], d_incoming[layer], d_gradC[layer], cweight_rows, cweight_cols);
                CU_CHECK(cudaGetLastError());

                // scale gradc by ALPHA
                scaleByValue<<<globalWeightGrad, local_1d>>>(d_gradC[layer], d_gradC[layer], ALPHA, (int)cweight_flat_size);
                CU_CHECK(cudaGetLastError());

                // dL/dB_l (Outer Product: ones x d_incoming[L])
                vecxvec2mat<<<globalWeightGrad, local_1d>>>(d_ones, d_incoming[layer], d_gradB[layer], cweight_rows, cweight_cols);
                CU_CHECK(cudaGetLastError());

                // scale gradb by 1-ALPHA
                scaleByValue<<<globalWeightGrad, local_1d>>>(d_gradB[layer], d_gradB[layer], 1.0f - ALPHA, (int)cweight_flat_size);
                CU_CHECK(cudaGetLastError());

                // --- Outgoing Gradient Calculation (for layer-1) ---
                dim3 globalTranspose = calculate_grid_2d(cweight_cols, cweight_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y);
                transpose<<<globalTranspose, local_2d>>>(d_cweights[layer], d_C_T, cweight_rows, cweight_cols);
                CU_CHECK(cudaGetLastError());

                // incoming gradient x C^T
                vecxmat2vec<<<globalOutGrad, local_1d>>>(d_incoming[layer], d_C_T, d_preoutgoing_l, cweight_cols, cweight_rows);
                CU_CHECK(cudaGetLastError());

                // derivative of power
                dPower<<<globalOutGrad, local_1d>>>(d_activate[layer - 1], d_dpow_l, order, prev_size);
                CU_CHECK(cudaGetLastError());

                // derivative of activation
                sigmoidDer<<<globalOutGrad, local_1d>>>(d_dotProds[layer - 1], d_dact_l, prev_size);
                CU_CHECK(cudaGetLastError());

                // outgoing gradient = (dl/dz_l x C^T) . dprev_p . dprevAct
                hadamard2<<<globalOutGrad, local_1d>>>(d_preoutgoing_l, d_dpow_l, d_dact_l, d_incoming[layer - 1], 1, prev_size);
                CU_CHECK(cudaGetLastError());
            }

            // Backpropagation for the first layer (input layer)
            int prev_size = input.size();
            int curr_size = activate[0].size();
            int cweight_rows = prev_size;
            int cweight_cols = curr_size;
            size_t cweight_flat_size = cweight_rows * cweight_cols;
            dim3 globalWeightGradFirst = calculate_grid_1d(cweight_flat_size, WORKSIZE_1D);

            // dL/dC_0 (Outer Product: d_in x d_incoming[0])
            vecxvec2mat<<<globalWeightGradFirst, local_1d>>>(d_in, d_incoming[0], d_gradC[0], cweight_rows, cweight_cols);
            CU_CHECK(cudaGetLastError());

            // scale gradc by ALPHA
            scaleByValue<<<globalWeightGradFirst, local_1d>>>(d_gradC[0], d_gradC[0], ALPHA, (int)cweight_flat_size);
            CU_CHECK(cudaGetLastError());

            // dL/dB_0 (Outer Product: ones x d_incoming[0])
            vecxvec2mat<<<globalWeightGradFirst, local_1d>>>(d_ones, d_incoming[0], d_gradB[0], cweight_rows, cweight_cols);
            CU_CHECK(cudaGetLastError());

            // scale gradb by 1-ALPHA
            scaleByValue<<<globalWeightGradFirst, local_1d>>>(d_gradB[0], d_gradB[0], 1.0f - ALPHA, (int)cweight_flat_size);
            CU_CHECK(cudaGetLastError());
            CU_CHECK(cudaDeviceSynchronize());

            // Update weights
            for (int i = 0; i < this->layers; ++i) {
                size_t c_size = cweights[i].size() * cweights[i][0].size();
                size_t b_size = bweights[i].size() * bweights[i][0].size();
                dim3 globalUpdate = calculate_grid_1d(c_size, WORKSIZE_1D);
                dim3 globalUpdateB = calculate_grid_1d(b_size, WORKSIZE_1D);

                switch (weightUpdateType) {
                    case 0:
                        kernelUpdateWeights<<<globalUpdate, local_1d>>>(d_cweights[i], d_gradC[i], learningRate, (int)c_size);
                        kernelUpdateWeights<<<globalUpdateB, local_1d>>>(d_bweights[i], d_gradB[i], learningRate, (int)b_size);
                        break;
                    case 1:
                        kernelUpdateWeightsWithL1<<<globalUpdate, local_1d>>>(d_cweights[i], d_gradC[i], (int)c_size, learningRate, LAMBDA_L1);
                        kernelUpdateWeightsWithL1<<<globalUpdateB, local_1d>>>(d_bweights[i], d_gradB[i], (int)b_size, learningRate, LAMBDA_L1);
                        break;
                    case 2:
                        kernelUpdateWeightsWithL2<<<globalUpdate, local_1d>>>(d_cweights[i], d_gradC[i], (int)c_size, learningRate, LAMBDA_L2);
                        kernelUpdateWeightsWithL2<<<globalUpdateB, local_1d>>>(d_bweights[i], d_gradB[i], (int)b_size, learningRate, LAMBDA_L2);
                        break;
                    case 3:
                        kernelUpdateWeightsElasticNet<<<globalUpdate, local_1d>>>(d_cweights[i], d_gradC[i], (int)c_size, learningRate, LAMBDA_L1, LAMBDA_L2);
                        kernelUpdateWeightsElasticNet<<<globalUpdateB, local_1d>>>(d_bweights[i], d_gradB[i], (int)b_size, learningRate, LAMBDA_L1, LAMBDA_L2);
                        break;
                    case 4:
                        kernelUpdateWeightsWithWeightDecay<<<globalUpdate, local_1d>>>(d_cweights[i], d_gradC[i], (int)c_size, learningRate, WEIGHT_DECAY);
                        kernelUpdateWeightsWithWeightDecay<<<globalUpdateB, local_1d>>>(d_bweights[i], d_gradB[i], (int)b_size, learningRate, WEIGHT_DECAY);
                        break;
                    case 5:
                        kernelUpdateWeightsDropout<<<globalUpdate, local_1d>>>(d_cweights[i], d_gradC[i], (int)c_size, learningRate, DROPOUT_RATE, (uint)rand());
                        kernelUpdateWeightsDropout<<<globalUpdateB, local_1d>>>(d_bweights[i], d_gradB[i], (int)b_size, learningRate, DROPOUT_RATE, (uint)rand());
                        break;
                }
            }
        }

        // Copy updated weights D2H
        for (int i = 0; i < layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            std::vector<float> c_upd(c_size), b_upd(b_size);
            CU_CHECK(cudaMemcpy(c_upd.data(), d_cweights[i], c_size * sizeof(float), cudaMemcpyDeviceToHost));
            CU_CHECK(cudaMemcpy(b_upd.data(), d_bweights[i], b_size * sizeof(float), cudaMemcpyDeviceToHost));
            cweights[i] = reshape(c_upd, cweights[i].size(), cweights[i][0].size());
            bweights[i] = reshape(b_upd, bweights[i].size(), bweights[i][0].size());
        }

        // --- Buffer Cleanup ---
        cudaFree(d_in); cudaFree(d_exp); cudaFree(d_out); cudaFree(d_err); cudaFree(d_ones);
        if (layers > 1) {
            cudaFree(d_preoutgoing_l); cudaFree(d_dpow_l); cudaFree(d_dact_l); cudaFree(d_C_T);
        }
        for (int i = 0; i < layers; ++i) {
            cudaFree(d_cweights[i]); cudaFree(d_bweights[i]);
            cudaFree(d_gradC[i]); cudaFree(d_gradB[i]);
            cudaFree(d_dotProds[i]); cudaFree(d_activate[i]);
            cudaFree(d_incoming[i]);
        }
    }
}

#endif // USE_CU