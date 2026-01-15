#ifdef USE_CU
#include "mnn.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cstdlib> // For rand()

/**
 * @brief Trains the mnn network on a single input-target pair with optimized buffer management.
 * @param input The input vector.
 * @param target The target output vector.
 */
void mnn::cuBufTrain(const std::vector<float>& input, const std::vector<float>& target) {
    // --- Buffer Allocation ---
    float *d_in = nullptr, *d_exp = nullptr, *d_out = nullptr, *d_err = nullptr, *d_ones = nullptr, *d_final = nullptr;
    std::vector<float*> d_cweights(layers), d_bweights(layers), d_gradC(layers), d_gradB(layers);
    std::vector<float*> d_dotProds(layers), d_activate(layers), d_incoming(layers);
    std::vector<float*> d_dpow(layers > 1 ? layers - 1 : 0), d_dact(layers > 1 ? layers - 1 : 0);
    float *d_preoutgoing_l = nullptr;

    try {
        // Allocate input/output/target buffers
        CU_CHECK(cudaMalloc(&d_in, input.size() * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_exp, target.size() * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_out, output.size() * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_final, output.size() * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_err, output.size() * sizeof(float)));
        CU_CHECK(cudaMemcpy(d_in, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
        CU_CHECK(cudaMemcpy(d_exp, target.data(), target.size() * sizeof(float), cudaMemcpyHostToDevice));

        size_t max_layer_width = 0;
        for (int w : width) max_layer_width = std::max(max_layer_width, (size_t)w);
        max_layer_width = std::max(max_layer_width, output.size());
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
            size_t max_outgoing_size = 0;
            for (int i = 0; i < layers - 1; i++) max_outgoing_size = std::max(max_outgoing_size, activate[i].size());
            if (max_outgoing_size > 0) {
                CU_CHECK(cudaMalloc(&d_preoutgoing_l, sizeof(float) * max_outgoing_size));
            }
            for (int i = 0; i < layers - 1; ++i) {
                size_t act_size = activate[i].size();
                CU_CHECK(cudaMalloc(&d_dpow[i], act_size * sizeof(float)));
                CU_CHECK(cudaMalloc(&d_dact[i], act_size * sizeof(float)));
            }
        }

        int epoch = 0;
        float initialLR = this->learningRate;

        // --- Training Loop ---
        while (true) {
            // Copy weights H2D for current iteration
            for (int i = 0; i < layers; ++i) {
                std::vector<float> flat_c = flatten(cweights[i]);
                std::vector<float> flat_b = flatten(bweights[i]);
                CU_CHECK(cudaMemcpy(d_cweights[i], flat_c.data(), flat_c.size() * sizeof(float), cudaMemcpyHostToDevice));
                CU_CHECK(cudaMemcpy(d_bweights[i], flat_b.data(), flat_b.size() * sizeof(float), cudaMemcpyHostToDevice));
            }

            // --- Forward Propagation ---
            float* current_input_ptr;
            // forprop on layers
            for(int i = 0; i < this->layers; i++) {
                if (i == 0) {
                    current_input_ptr = d_in;
                } else {
                    current_input_ptr = d_activate[i-1];
                }

                int current_in_size = (i == 0) ? (int)input.size() : (int)activate[i-1].size();
                int current_out_size = (int)dotProds[i].size();
                dim3 block_1d(WORKSIZE_1D);
                dim3 grid_forward = calculate_grid_1d(current_out_size, WORKSIZE_1D);

                kernelLayerForward2<<<grid_forward, block_1d>>>(
                    current_input_ptr,
                    d_dotProds[i],
                    d_cweights[i],
                    d_bweights[i],
                    current_in_size,
                    current_out_size,
                    order
                );

                dim3 grid_sigmoid = calculate_grid_1d(current_out_size, WORKSIZE_1D);
                sigmoid<<<grid_sigmoid, block_1d>>>(
                    d_dotProds[i],
                    d_activate[i],
                    current_out_size
                );
            }
            CU_CHECK(cudaDeviceSynchronize());

            // Copy output D2H to check for correctness and loss
            CU_CHECK(cudaMemcpy(activate[layers-1].data(), d_final, activate[layers-1].size() * sizeof(float), cudaMemcpyDeviceToHost));
            CU_CHECK(cudaMemcpy(output.data(), d_out, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
            output = softmax(output, SOFTMAX_TEMP);

            if (maxIndex(output) == maxIndex(target)) {
                std::cout << "Correct output predicted at epoch " << epoch << " with loss " << crossEntropy(output, target) << "." << std::endl;
                break;
            }
            epoch++;
            currloss = crossEntropy(output, target);
            // std::cout << "Current CE Loss at epoch " << epoch << " : " << currloss << std::endl;

            // --- Backward Propagation ---
            dim3 block_1d(WORKSIZE_1D);
            dim3 block_2d(WORKSIZE_2D_X, WORKSIZE_2D_Y);
            size_t outputSize = output.size();

            dim3 gridSub = calculate_grid_1d(outputSize, WORKSIZE_1D);
            subtract<<<gridSub, block_1d>>>(d_out, d_exp, d_err, (int)outputSize);
            CU_CHECK(cudaMemcpy(d_incoming[layers - 1], d_err, sizeof(float) * outputSize, cudaMemcpyDeviceToDevice));

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

                // --- Outgoing Gradient Calculation (for layer-1) ---
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

                // derivative of power
                dPower<<<gridOutGrad, block_1d>>>(
                    d_activate[layer - 1], d_dpow[layer - 1], order, prev_size
                );
                // derivative of activation
                sigmoidDer<<<gridOutGrad, block_1d>>>(
                    d_dotProds[layer - 1], d_dact[layer - 1], prev_size
                );
                // outgoing gradient = (dl/dz_l x C^T) . dprev_p . dprevAct
                hadamard2<<<gridOutGrad, block_1d>>>(
                    d_preoutgoing_l, d_dpow[layer - 1], d_dact[layer - 1], d_incoming[layer - 1], 1, prev_size
                );
            }

            // Backpropagation for the first layer (input layer)
            int prev_size_first = input.size();
            int curr_size = activate[0].size();
            int cweight_rows = prev_size_first;
            int cweight_cols = curr_size;
            size_t cweight_flat_size = cweight_rows * cweight_cols;
            dim3 gridWeightGrad = calculate_grid_1d(cweight_flat_size, WORKSIZE_1D);

            // dL/dC_0 (Outer Product: d_in x d_incoming[0])
            vecxvec2mat<<<gridWeightGrad, block_1d>>>(
                d_in, d_incoming[0], d_gradC[0], cweight_rows, cweight_cols
            );
            // scale gradc by ALPHA
            scaleByValue<<<gridWeightGrad, block_1d>>>(
                d_gradC[0], d_gradC[0], ALPHA, (int)cweight_flat_size
            );

            // dL/dB_0 (Outer Product: ones x d_incoming[0])
            vecxvec2mat<<<gridWeightGrad, block_1d>>>(
                d_ones, d_incoming[0], d_gradB[0], cweight_rows, cweight_cols
            );
            // scale gradb by 1-ALPHA
            scaleByValue<<<gridWeightGrad, block_1d>>>(
                d_gradB[0], d_gradB[0], 1.0f - ALPHA, (int)cweight_flat_size
            );

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

            }
            CU_CHECK(cudaDeviceSynchronize());
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

        this->learningRate = initialLR;
        std::cout << "Training complete for this input-target pair." << std::endl;

        // --- Buffer Cleanup ---
        cudaFree(d_in); cudaFree(d_exp); cudaFree(d_out); cudaFree(d_err); cudaFree(d_ones);
        for (int i = 0; i < layers; ++i) {
            cudaFree(d_cweights[i]); cudaFree(d_bweights[i]);
            cudaFree(d_gradC[i]); cudaFree(d_gradB[i]);
            cudaFree(d_dotProds[i]); cudaFree(d_activate[i]);
            cudaFree(d_incoming[i]);
            if (i < layers - 1) {
                cudaFree(d_dpow[i]); cudaFree(d_dact[i]);
            }
        }
        cudaFree(d_preoutgoing_l);

    }
    catch (const std::runtime_error& e) {
        std::cerr << "Error during cuBufTrain: " << e.what() << std::endl;
    }
}

/**
 * @brief Trains the mnn2d network on a single input-target pair with optimized buffer management.
 * @param input The input matrix.
 * @param target The target vector.
 */
void mnn2d::cuBufTrain(const std::vector<std::vector<float>>& input, const std::vector<float>& target) {
    float *d_in = nullptr, *d_exp = nullptr, *d_out = nullptr, *d_err = nullptr, *d_final = nullptr;
    std::vector<float*> d_cweights(layers), d_bweights(layers), d_gradC(layers), d_gradB(layers);
    std::vector<float*> d_dotProds(layers), d_activate(layers), d_incoming(layers);
    float *d_grad_x_CT = nullptr, *d_dprev_p = nullptr, *d_dprev_act = nullptr;

    try {
        dim3 block_1d(WORKSIZE_1D);
        dim3 block_2d(WORKSIZE_2D_X, WORKSIZE_2D_Y);

        // --- Buffer Allocation ---
        std::vector<float> flat_input = flatten(input);
        CU_CHECK(cudaMalloc(&d_in, flat_input.size() * sizeof(float)));
        CU_CHECK(cudaMemcpy(d_in, flat_input.data(), flat_input.size() * sizeof(float), cudaMemcpyHostToDevice));
        CU_CHECK(cudaMalloc(&d_exp, target.size() * sizeof(float)));
        CU_CHECK(cudaMemcpy(d_exp, target.data(), target.size() * sizeof(float), cudaMemcpyHostToDevice));
        CU_CHECK(cudaMalloc(&d_out, output.size() * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_final, output.size() * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_err, output.size() * sizeof(float)));

        size_t max_act_size = 0;
        for (int i = 0; i < layers; ++i) max_act_size = std::max(max_act_size, activate[i].size() * activate[i][0].size());
        max_act_size = std::max(max_act_size, (size_t)inHeight * inWidth);
        CU_CHECK(cudaMalloc(&d_grad_x_CT, sizeof(float) * max_act_size));
        CU_CHECK(cudaMalloc(&d_dprev_p, sizeof(float) * max_act_size));
        CU_CHECK(cudaMalloc(&d_dprev_act, sizeof(float) * max_act_size));

        for (int i = 0; i < layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            size_t act_size = activate[i].size() * activate[i][0].size();
            CU_CHECK(cudaMalloc(&d_cweights[i], c_size * sizeof(float))); CU_CHECK(cudaMalloc(&d_bweights[i], b_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_gradC[i], c_size * sizeof(float))); CU_CHECK(cudaMalloc(&d_gradB[i], b_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_dotProds[i], act_size * sizeof(float))); CU_CHECK(cudaMalloc(&d_activate[i], act_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_incoming[i], act_size * sizeof(float)));
        }

        int epoch = 0;
        float initialLR = this->learningRate;

        // --- Training Loop ---
        while (true) {
            for (int i = 0; i < layers; ++i) { // Copy weights H2D
                std::vector<float> flat_c = flatten(cweights[i]); std::vector<float> flat_b = flatten(bweights[i]);
                CU_CHECK(cudaMemcpy(d_cweights[i], flat_c.data(), flat_c.size() * sizeof(float), cudaMemcpyHostToDevice));
                CU_CHECK(cudaMemcpy(d_bweights[i], flat_b.data(), flat_b.size() * sizeof(float), cudaMemcpyHostToDevice));
            }

            // --- Forward Propagation ---
            float* d_current_act = d_in;
            for (int i = 0; i < layers; ++i) {
                int currentInHeight = (i == 0) ? inHeight : activate[i-1].size();
                int currentInWidth = (i == 0) ? inWidth : activate[i-1][0].size();
                int currentoutSize = width[i];
                dim3 grid_forward = calculate_grid_2d(currentoutSize, currentInHeight, WORKSIZE_2D_X, WORKSIZE_2D_Y);
                kernelLayerForward4<<<grid_forward, block_2d>>>(
                    d_current_act,
                    d_dotProds[i],
                    d_cweights[i],
                    d_bweights[i],
                    currentInHeight,
                    currentInWidth,
                    currentoutSize,
                    order
                );

                size_t dotprod_size = currentInHeight * currentoutSize;
                relu<<<calculate_grid_1d(dotprod_size, WORKSIZE_1D), block_1d>>>(
                    d_dotProds[i],
                    d_activate[i],
                    dotprod_size
                );
                d_current_act = d_activate[i];
            }

            meanPool<<<calculate_grid_1d(outSize, WORKSIZE_1D), block_1d>>>(
                d_activate[layers - 1],
                d_out,
                inHeight,
                outSize,
                1
            );
            CU_CHECK(cudaDeviceSynchronize());
            CU_CHECK(cudaMemcpy(output.data(), d_out, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
            output = softmax(output, SOFTMAX_TEMP);

            if (maxIndex(output) == maxIndex(target)) {
                std::cout << "Correct output predicted at epoch " << epoch << "." << std::endl;
                break;
            }
            epoch++;
            currloss = crossEntropy(output, target);
            // std::cout << "Current CE Loss at epoch " << epoch << " : " << currloss << std::endl;

            // --- Backward Propagation ---
            // Initial error (output - expected)
            subtract<<<calculate_grid_1d(outSize, WORKSIZE_1D), block_1d>>>(
                d_out,  // mean pooled, not softmaxed
                d_exp,
                d_err,
                (int)outSize
            );
            scaleByValue<<<calculate_grid_1d(outSize, WORKSIZE_1D), block_1d>>>(
                d_err,
                d_err,
                1.0f / (float)activate[layers - 1].size(),
                (int)outSize
            );
            // Distribute the error from d_err to each row of the last layer's incoming gradient buffer.
            for (size_t r = 0; r < activate[layers - 1].size(); ++r) {
                CU_CHECK(cudaMemcpy(d_incoming[layers - 1] + r * outSize, d_err, sizeof(float) * outSize, cudaMemcpyDeviceToDevice));
            }

            // Backpropagation loop (last layer to second layer)
            for (int layer = layers - 1; layer >= 1; --layer) {
                int prev_rows = activate[layer - 1].size(), prev_cols = activate[layer - 1][0].size();
                int curr_rows = activate[layer].size(), curr_cols = activate[layer][0].size();
                int cweight_rows = cweights[layer].size(), cweight_cols = cweights[layer][0].size();
                size_t cweight_flat_size = cweight_rows * cweight_cols;
                size_t prev_act_flat_size = prev_rows * prev_cols;

                // transpose
                float *d_C_T; CU_CHECK(cudaMalloc(&d_C_T, cweight_flat_size * sizeof(float)));
                transpose<<<calculate_grid_2d(cweight_cols, cweight_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                    d_cweights[layer], d_C_T, cweight_rows, cweight_cols
                );
                // dL/dz_l x C^T
                matxmat2mat<<<calculate_grid_2d(prev_cols, curr_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                    d_incoming[layer], d_C_T, d_grad_x_CT, curr_rows, curr_cols, prev_cols
                );

                // d(prev_p)
                dPower<<<calculate_grid_1d(prev_act_flat_size, WORKSIZE_1D), block_1d>>>(
                    d_activate[layer - 1], d_dprev_p, order, prev_act_flat_size
                );
                // Calculate d(prev_act)
                reluDer<<<calculate_grid_1d(prev_act_flat_size, WORKSIZE_1D), block_1d>>>(
                    d_dotProds[layer - 1], d_dprev_act, prev_act_flat_size
                );
                // outgoing = (dL/dz_l * C^T) . d(prev_p) . d(prev_act)
                hadamard2<<<calculate_grid_1d(prev_act_flat_size, WORKSIZE_1D), block_1d>>>(
                    d_grad_x_CT, d_dprev_p, d_dprev_act, d_incoming[layer - 1], prev_rows, prev_cols
                );

                // --- Calculate Weight Gradients ---
                // gradc = ALPHA * prev_p^T * incoming
                // power prev_p
                float *d_prev_p, *d_prev_p_T;
                CU_CHECK(cudaMalloc(&d_prev_p, prev_act_flat_size * sizeof(float)));
                CU_CHECK(cudaMalloc(&d_prev_p_T, prev_act_flat_size * sizeof(float)));
                power<<<calculate_grid_1d(prev_act_flat_size, WORKSIZE_1D), block_1d>>>(
                    d_activate[layer - 1], d_prev_p, order, prev_act_flat_size
                );
                // transpose prev_p
                transpose<<<calculate_grid_2d(prev_cols, prev_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                    d_prev_p, d_prev_p_T, prev_rows, prev_cols
                );

                // dL/dC_layer
                matxmat2mat<<<calculate_grid_2d(curr_cols, prev_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                    d_prev_p_T, d_incoming[layer], d_gradC[layer], prev_cols, prev_rows, curr_cols
                );
                // scale dL/dC_layer by ALPHA
                scaleByValue<<<calculate_grid_1d(cweight_flat_size, WORKSIZE_1D), block_1d>>>(
                    d_gradC[layer], d_gradC[layer], ALPHA, cweight_flat_size
                );

                // gradB = dL/dz_l x V1^T
                // transpose ones = onesT
                float* d_onesT; CU_CHECK(cudaMalloc(&d_onesT, prev_act_flat_size * sizeof(float)));
                std::vector<float> h_ones(prev_act_flat_size, 1.0f);
                CU_CHECK(cudaMemcpy(d_onesT, h_ones.data(), prev_act_flat_size * sizeof(float), cudaMemcpyHostToDevice));
                // dL/dB_layer
                matxmat2mat<<<calculate_grid_2d(curr_cols, prev_rows, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                    d_onesT, d_incoming[layer], d_gradB[layer], prev_cols, prev_rows, curr_cols
                );
                // scale dL/dB_layer by 1- ALPHA
                scaleByValue<<<calculate_grid_1d(cweight_flat_size, WORKSIZE_1D), block_1d>>>(
                    d_gradB[layer], d_gradB[layer], 1.0f - ALPHA, cweight_flat_size
                );

                cudaFree(d_C_T); cudaFree(d_prev_p); cudaFree(d_prev_p_T); cudaFree(d_onesT);
            }

            // Backprop for first layer (input layer)
            int in_h = input.size(), in_w = input[0].size();
            int first_layer_cols = activate[0][0].size();
            size_t cweight_flat_size_first = in_w * first_layer_cols;
            size_t input_flat_size = in_h * in_w;

            float *d_input_p, *d_input_p_T;
            CU_CHECK(cudaMalloc(&d_input_p, input_flat_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_input_p_T, input_flat_size * sizeof(float)));
            // power input
            power<<<calculate_grid_1d(input_flat_size, WORKSIZE_1D), block_1d>>>(
                d_in, d_input_p, order, input_flat_size
            );
            // transpose input
            transpose<<<calculate_grid_2d(in_w, in_h, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                d_input_p, d_input_p_T, in_h, in_w
            );

            // dL/dC_0
            matxmat2mat<<<calculate_grid_2d(first_layer_cols, in_h, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                d_input_p_T, d_incoming[0], d_gradC[0], in_w, in_h, first_layer_cols
            );
            // scale by ALPHA
            scaleByValue<<<calculate_grid_1d(cweight_flat_size_first, WORKSIZE_1D), block_1d>>>(
                d_gradC[0], d_gradC[0], ALPHA, cweight_flat_size_first
            );

            // dL/dB_0
            float* d_onesT_first; CU_CHECK(cudaMalloc(&d_onesT_first, input_flat_size * sizeof(float)));
            std::vector<float> h_ones_first(input_flat_size, 1.0f);
            CU_CHECK(cudaMemcpy(d_onesT_first, h_ones_first.data(), input_flat_size * sizeof(float), cudaMemcpyHostToDevice));
            matxmat2mat<<<calculate_grid_2d(first_layer_cols, in_h, WORKSIZE_2D_X, WORKSIZE_2D_Y), block_2d>>>(
                d_onesT_first, d_incoming[0], d_gradB[0], in_w, in_h, first_layer_cols
            );
            // scale by 1
            scaleByValue<<<calculate_grid_1d(cweight_flat_size_first, WORKSIZE_1D), block_1d>>>(
                d_gradB[0], d_gradB[0], 1.0f - ALPHA, cweight_flat_size_first
            );

            cudaFree(d_input_p); cudaFree(d_input_p_T); cudaFree(d_onesT_first);

            // --- Update Weights ---
            for (int i = 0; i < layers; ++i) {
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
            }
            CU_CHECK(cudaDeviceSynchronize());
        }

        this->learningRate = initialLR;
        // Copy final weights back to host
        for (int i = 0; i < layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            std::vector<float> c_upd(c_size), b_upd(b_size);
            CU_CHECK(cudaMemcpy(c_upd.data(), d_cweights[i], c_size * sizeof(float), cudaMemcpyDeviceToHost));
            CU_CHECK(cudaMemcpy(b_upd.data(), d_bweights[i], b_size * sizeof(float), cudaMemcpyDeviceToHost));
            cweights[i] = reshape(c_upd, cweights[i].size(), cweights[i][0].size());
            bweights[i] = reshape(b_upd, bweights[i].size(), bweights[i][0].size());
        }
        std::cout << "Training complete for this input-target pair." << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "Error during cuBufTrain (mnn2d): " << e.what() << std::endl;
    }

    // --- Final Buffer Cleanup ---
    cudaFree(d_in); cudaFree(d_exp); cudaFree(d_out); cudaFree(d_err);
    cudaFree(d_grad_x_CT); cudaFree(d_dprev_p); cudaFree(d_dprev_act);
    for (int i = 0; i < layers; ++i) {
        cudaFree(d_cweights[i]); cudaFree(d_bweights[i]);
        cudaFree(d_gradC[i]); cudaFree(d_gradB[i]);
        cudaFree(d_dotProds[i]); cudaFree(d_activate[i]);
        cudaFree(d_incoming[i]);
    }
}

#endif // USE_CU