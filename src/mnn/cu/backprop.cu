#ifdef USE_CUDA
#include "mnn.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm> // For std::copy
#include <cmath> // For std::ceil

/**
 * @brief Backpropagation for mnn using CUDA.
 * @param expected The expected output vector.
 */
void mnn::cuBackprop(const std::vector<float>& expected) {
    try {
        dim3 block_1d(CUDA_BLOCK_SIZE_1D);
        dim3 block_2d(CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y);
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

        dim3 gridSub = calculate_grid_1d(outputSize, CUDA_BLOCK_SIZE_1D);
        subtract<<<gridSub, block_1d>>>(d_out, d_exp, d_err, (int)outputSize);
        
        for(int i = 0; i < this->layers; i++) {
            std::vector<float> flat_c = flatten(cweights[i]);
            std::vector<float> flat_b = flatten(bweights[i]);
            size_t cweight_size = flat_c.size();
            size_t bweight_size = flat_b.size();
            size_t act_size = activate[i].size();

            CUDA_CHECK(cudaMalloc(&d_cweights[i], sizeof(float) * cweight_size));
            CUDA_CHECK(cudaMalloc(&d_bweights[i], sizeof(float) * bweight_size));
            CUDA_CHECK(cudaMemcpy(d_cweights[i], flat_c.data(), sizeof(float) * cweight_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_bweights[i], flat_b.data(), sizeof(float) * bweight_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_gradC[i], sizeof(float) * cweight_size));
            CUDA_CHECK(cudaMalloc(&d_gradB[i], sizeof(float) * bweight_size));
            CUDA_CHECK(cudaMalloc(&d_activate[i], sizeof(float) * act_size));
            CUDA_CHECK(cudaMalloc(&d_dotProds[i], sizeof(float) * act_size));
            CUDA_CHECK(cudaMemcpy(d_activate[i], activate[i].data(), sizeof(float) * act_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_dotProds[i], dotProds[i].data(), sizeof(float) * act_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_incoming[i], sizeof(float) * act_size));
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


        for(int layer = layers - 1; layer >= 1; layer--) {
            int prev_size = activate[layer - 1].size();
            int curr_size = activate[layer].size(); // Size of d_incoming[layer]
            int cweight_rows = prev_size;
            int cweight_cols = curr_size;
            size_t cweight_flat_size = cweight_rows * cweight_cols;

            dim3 gridWeightGrad = calculate_grid_1d(cweight_flat_size, CUDA_BLOCK_SIZE_1D);
            dim3 gridOutGrad = calculate_grid_1d(prev_size, CUDA_BLOCK_SIZE_1D);

            // dL/dC_l (Outer Product: d_incoming[L] x d_activate[L-1])
            vecxvec2mat<<<gridWeightGrad, block_1d>>>(
                d_activate[layer - 1],
                d_incoming[layer],
                d_gradC[layer],
                cweight_rows,
                cweight_cols
            );
            scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradC[layer], d_gradC[layer], alpha, (int)cweight_flat_size);

            // dL/dB_l (Outer Product: ones x d_incoming[L])
            vecxvec2mat<<<gridWeightGrad, block_1d>>>(
                d_ones,
                d_incoming[layer],
                d_gradB[layer],
                cweight_rows,
                cweight_cols
            );
            scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradB[layer], d_gradB[layer], alpha, (int)cweight_flat_size);

            // Outgoing Gradient Calculation (for layer-1)
            float *d_C_T = nullptr;
            CUDA_CHECK(cudaMalloc(&d_C_T, sizeof(float) * cweight_flat_size));
            transpose<<<calculate_grid_2d(cweight_cols, cweight_rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), block_2d>>>(
                d_cweights[layer],
                d_C_T,
                cweight_rows, // rows
                cweight_cols  // cols
            );

            // incoming gradient x C^T
            vecxmat2vec<<<gridOutGrad, block_1d>>>(
                d_incoming[layer],
                d_C_T,
                d_preoutgoing_l,
                cweight_cols,
                cweight_rows
            );
            CUDA_CHECK(cudaFree(d_C_T));

            dPower<<<gridOutGrad, block_1d>>>(d_activate[layer - 1], d_dpow[layer - 1], order, prev_size);
            sigmoidDer<<<gridOutGrad, block_1d>>>(d_dotProds[layer - 1], d_dact[layer - 1], prev_size);
            hadamard2<<<gridOutGrad, block_1d>>>(
                d_preoutgoing_l,
                d_dpow[layer - 1],
                d_dact[layer - 1],
                d_incoming[layer - 1],
                1,
                prev_size
            );
        }

        int prev_size = input.size();
        int curr_size = activate[0].size();
        int cweight_rows = prev_size;
        int cweight_cols = curr_size;
        size_t cweight_flat_size = cweight_rows * cweight_cols;
        dim3 gridWeightGrad = calculate_grid_1d(cweight_flat_size, CUDA_BLOCK_SIZE_1D);

        // 1. dL/dC_1 (Outer Product: d_incoming[0] x d_in)
        vecxvec2mat<<<gridWeightGrad, block_1d>>>(
            d_in,
            d_incoming[0],
            d_gradC[0],
            cweight_rows,
            cweight_cols
        );
        scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradC[0], d_gradC[0], alpha, (int)cweight_flat_size);

        vecxvec2mat<<<gridWeightGrad, block_1d>>>(
            d_ones,
            d_incoming[0],
            d_gradB[0],
            cweight_rows,
            cweight_cols
        );
        scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradB[0], d_gradB[0], alpha, (int)cweight_flat_size);

        // --- Update Weights and Copy Results Back ---
        for (int i = 0; i < this->layers; ++i) {
            std::vector<float> flat_c = flatten(cweights[i]);
            std::vector<float> flat_b = flatten(bweights[i]);
            size_t c_size = flat_c.size();
            size_t b_size = flat_b.size();
            dim3 gridUpdate = calculate_grid_1d(c_size, CUDA_BLOCK_SIZE_1D);

            kernelUpdateWeightsElasticNet<<<gridUpdate, block_1d>>>(d_cweights[i], d_gradC[i], (int)c_size,
                                            learningRate, LAMBDA_L1, LAMBDA_L2);
            kernelUpdateWeightsElasticNet<<<gridUpdate, block_1d>>>(d_bweights[i], d_gradB[i], (int)b_size,
                                            learningRate, LAMBDA_L1, LAMBDA_L2);

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
        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_exp)); CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_err)); CUDA_CHECK(cudaFree(d_ones));
        CUDA_CHECK(cudaFree(d_preoutgoing_l)); CUDA_CHECK(cudaFree(d_outgoing_l));
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
void mnn2d::cuBackprop(const std::vector<float>& expected) {
    // --- Variable Declarations (moved to top of try block for explicit cleanup) ---
    float *d_in = nullptr, *d_exp = nullptr, *d_out = nullptr, *d_err = nullptr;
    float *d_grad_x_CT = nullptr; // Reusable global temporary
    
    std::vector<float*> d_incoming(this->layers, nullptr);
    std::vector<float*> d_cweights(this->layers, nullptr);
    std::vector<float*> d_bweights(this->layers, nullptr);
    std::vector<float*> d_gradC(this->layers, nullptr);
    std::vector<float*> d_gradB(this->layers, nullptr);
    std::vector<float*> d_dotProds(this->layers, nullptr);
    std::vector<float*> d_activate(this->layers, nullptr);
    std::vector<float*> d_outgoing(this->layers - 1, nullptr);
    std::vector<float*> d_dpow(this->layers - 1, nullptr); // Used for d(prev_p)
    std::vector<float*> d_dact(this->layers - 1, nullptr); // Used for d(prev_act)

    // TEMPORARY POINTERS FOR LOOP AND FIRST LAYER (initialized to nullptr for safety)
    float *d_C_T = nullptr, *d_prev_p = nullptr, *d_prev_p_T = nullptr;
    float *d_input_p = nullptr, *d_input_p_T = nullptr;
    float *d_ones = nullptr, *d_onesT = nullptr;
    
    size_t max_act_size = 0;
    
    try {
        dim3 block_1d(CUDA_BLOCK_SIZE_1D);
        dim3 block_2d(CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y);

        std::vector<float> flat_input = flatten(input);
        size_t input_flat_size = flat_input.size();
        size_t outputSize = output.size();
        CUDA_CHECK(cudaMalloc(&d_in, sizeof(float) * input_flat_size));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * outputSize));
        CUDA_CHECK(cudaMalloc(&d_exp, sizeof(float) * outputSize));
        CUDA_CHECK(cudaMalloc(&d_err, sizeof(float) * outputSize));
        CUDA_CHECK(cudaMemcpy(d_in, flat_input.data(), sizeof(float) * input_flat_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out, output.data(), sizeof(float) * outputSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_exp, expected.data(), sizeof(float) * outputSize, cudaMemcpyHostToDevice));

        // --- Layer-Specific Allocations ---
        for (int i = 0; i < this->layers; ++i) {
            std::vector<float> flat_c = flatten(cweights[i]);
            std::vector<float> flat_b = flatten(bweights[i]);
            size_t c_size = flat_c.size();
            size_t b_size = flat_b.size();
            size_t dot_size = dotProds[i].size() * dotProds[i][0].size();
            max_act_size = std::max(max_act_size, dot_size); // Calculate max_act_size

            CUDA_CHECK(cudaMalloc(&d_cweights[i], c_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_bweights[i], b_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_dotProds[i], dot_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_activate[i], dot_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_gradC[i], c_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_gradB[i], b_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_incoming[i], dot_size * sizeof(float)));

            CUDA_CHECK(cudaMemcpy(d_cweights[i], flat_c.data(), c_size * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_bweights[i], flat_b.data(), b_size * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_dotProds[i], flatten(dotProds[i]).data(), dot_size * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_activate[i], flatten(activate[i]).data(), dot_size * sizeof(float), cudaMemcpyHostToDevice));
        }
        for (int i = 0; i < this->layers - 1; ++i) {
            size_t act_size = activate[i].size() * activate[i][0].size();
            CUDA_CHECK(cudaMalloc(&d_outgoing[i], act_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_dpow[i], act_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_dact[i], act_size * sizeof(float)));
        }

        // --- Allocate Reusable/Global Temporary Buffer for dL/dz * C^T ---
        if (max_act_size > 0) {
            CUDA_CHECK(cudaMalloc(&d_grad_x_CT, max_act_size * sizeof(float)));
        }

        // Initial error (output - expected)
        dim3 gridSub = calculate_grid_1d(outputSize, CUDA_BLOCK_SIZE_1D);
        subtract<<<gridSub, block_1d>>>(d_out, d_exp, d_err, (int)outputSize);

        std::vector<float> out_err_host(outputSize);
        CUDA_CHECK(cudaMemcpy(out_err_host.data(), d_err, sizeof(float) * outputSize, cudaMemcpyDeviceToHost));
        
        int last_rows = activate[layers - 1].size();
        int last_cols = activate[layers - 1][0].size();
        std::vector<float> last_layer_err(last_rows * last_cols);
        for(int r = 0; r < last_rows; ++r) {
            for(int c = 0; c < last_cols; ++c) {
                last_layer_err[r * last_cols + c] = out_err_host[c] / (float)last_rows;
            }
        }
        CUDA_CHECK(cudaMemcpy(d_incoming[layers - 1], last_layer_err.data(), last_layer_err.size() * sizeof(float), cudaMemcpyHostToDevice));

        // --- Backpropagation Loop (Last Layer to Second Layer) ---
        for (int layer = layers - 1; layer >= 1; --layer) {
            int prev_rows = activate[layer-1].size();
            int prev_cols = activate[layer-1][0].size();
            int curr_rows = activate[layer].size();
            int curr_cols = activate[layer][0].size();
            int cweight_rows = prev_cols;
            int cweight_cols = curr_cols;
            size_t prev_act_size = prev_rows * prev_cols;
            size_t cweight_flat_size = cweight_rows * cweight_cols;

            // Transpose C: C_P_C -> C_C_P
            d_C_T = nullptr; // Reset for new allocation
            CUDA_CHECK(cudaMalloc(&d_C_T, sizeof(float) * cweight_flat_size));
            transpose<<<calculate_grid_2d(cweight_cols, cweight_rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), block_2d>>>(
                d_cweights[layer], d_C_T, cweight_rows, cweight_cols
            );

            // grad x C^T: (R_curr x C_curr) * (C_curr x P_cols) -> R_curr x P_cols
            dim3 gridMatMul = calculate_grid_2d(prev_cols, curr_rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y);
            matxmat2mat<<<gridMatMul, block_2d>>>(
                d_incoming[layer], d_C_T, d_grad_x_CT, curr_rows, curr_cols, prev_cols
            );

            // d(prev_p) - Use d_dpow[layer-1]
            dim3 grid1D_prev = calculate_grid_1d(prev_act_size, CUDA_BLOCK_SIZE_1D);
            dPower<<<grid1D_prev, block_1d>>>(d_activate[layer-1], d_dpow[layer-1], order, (int)prev_act_size);

            // d(prev_act) - Use d_dact[layer-1]
            softmaxDer<<<dim3(1), dim3(prev_act_size)>>>(d_dotProds[layer-1], d_dact[layer-1], SOFTMAX_TEMP, (int)prev_act_size);

            // outgoing = (dL/dz * C^T) . d(prev_p) . d(prev_act)
            hadamard2<<<grid1D_prev, block_1d>>>(
                d_grad_x_CT, d_dpow[layer-1], d_dact[layer-1], d_outgoing[layer-1], prev_rows, prev_cols
            );

            // a_prev -> a_prev^n
            d_prev_p = nullptr; // Reset for new allocation
            CUDA_CHECK(cudaMalloc(&d_prev_p, sizeof(float) * prev_act_size));
            power<<<grid1D_prev, block_1d>>>(d_activate[layer-1], d_prev_p, order, (int)prev_act_size);

            // Transpose prev_p: (R_prev x P_cols) -> (P_cols x R_prev)
            d_prev_p_T = nullptr; // Reset for new allocation
            size_t prev_p_T_size = prev_cols * prev_rows;
            CUDA_CHECK(cudaMalloc(&d_prev_p_T, sizeof(float) * prev_p_T_size));
            transpose<<<calculate_grid_2d(prev_cols, prev_rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), block_2d>>>(
                d_prev_p, d_prev_p_T, prev_rows, prev_cols
            );

            // dL/dC_layer: (P_cols x R_prev) * (R_curr x C_curr) -> P_cols x C_curr
            gridMatMul = calculate_grid_2d(cweight_cols, cweight_rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y);
            matxmat2mat<<<gridMatMul, block_2d>>>(
                d_prev_p_T, d_incoming[layer], d_gradC[layer], cweight_rows, curr_rows, cweight_cols
            );

            scaleByValue<<<calculate_grid_1d(cweight_flat_size, CUDA_BLOCK_SIZE_1D), block_1d>>>(
                d_gradC[layer], d_gradC[layer], alpha, (int)cweight_flat_size
            );

            // dL/dB_layer: (P_cols x R) * (R x C_curr) -> P_cols x C_curr
            d_ones = nullptr; d_onesT = nullptr;           
            CUDA_CHECK(cudaMalloc(&d_ones, sizeof(float) * prev_act_size));
            CUDA_CHECK(cudaMalloc(&d_onesT, sizeof(float) * prev_p_T_size));
            fill<<<calculate_grid_1d(prev_act_size, CUDA_BLOCK_SIZE_1D), block_1d>>>(d_ones, 1.0f, (int)prev_act_size);
            transpose<<<calculate_grid_2d(prev_cols, prev_rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y), block_2d>>>(
                d_ones, d_onesT, prev_rows, prev_cols
            );

            matxmat2mat<<<gridMatMul, block_2d>>>(
                d_onesT, d_incoming[layer], d_gradB[layer], cweight_rows, curr_rows, cweight_cols
            );

            size_t bweight_flat_size = bweights[layer].size() * bweights[layer][0].size();
            scaleByValue<<<calculate_grid_1d(bweight_flat_size, CUDA_BLOCK_SIZE_1D), block_1d>>>(
                d_gradB[layer], d_gradB[layer], 1.0f - alpha, (int)bweight_flat_size
            );

            CUDA_CHECK(cudaMemcpy(d_incoming[layer-1], d_outgoing[layer-1], sizeof(float) * prev_act_size, cudaMemcpyDeviceToDevice));
            
            // --- CLEANUP OF TEMPORARY POINTERS FOR THIS ITERATION ---
            if (d_prev_p) CUDA_CHECK(cudaFree(d_prev_p));
            if (d_prev_p_T) CUDA_CHECK(cudaFree(d_prev_p_T));
            if (d_ones) CUDA_CHECK(cudaFree(d_ones));
            if (d_C_T) CUDA_CHECK(cudaFree(d_C_T));
            if (d_onesT) CUDA_CHECK(cudaFree(d_onesT));
        }

        // Backpropagation for the First Layer (Layer 0)
        int inHeight = input.size();
        int inWidth = input[0].size();
        int firstLayerCols = activate[0][0].size();
        int cweight_rows = inWidth;
        int cweight_cols = firstLayerCols;
        size_t cweight_flat_size = cweight_rows * cweight_cols;
        size_t in_flat_size = inHeight * inWidth;

        // input^n: x -> x^n
        d_input_p = nullptr; // Reset for new allocation
        dim3 grid1D_in = calculate_grid_1d(in_flat_size, CUDA_BLOCK_SIZE_1D);
        CUDA_CHECK(cudaMalloc(&d_input_p, sizeof(float) * in_flat_size));
        power<<<grid1D_in, block_1d>>>(d_in, d_input_p, order, (int)in_flat_size);

        // Transpose x^n: (H x W) -> (W x H)
        d_input_p_T = nullptr; // Reset for new allocation
        size_t in_p_T_size = inWidth * inHeight;
        dim3 grid2D_in = calculate_grid_2d(inWidth, inHeight, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y);
        CUDA_CHECK(cudaMalloc(&d_input_p_T, sizeof(float) * in_p_T_size));
        transpose<<<grid2D_in, block_2d>>>(d_input_p, d_input_p_T, inHeight, inWidth);
        if (d_input_p) CUDA_CHECK(cudaFree(d_input_p));

        // dL/dC_1: (W x H) * (H x C_1) -> W x C_1
        dim3 gridMatMul = calculate_grid_2d(cweight_cols, cweight_rows, CUDA_BLOCK_SIZE_2D_X, CUDA_BLOCK_SIZE_2D_Y);
        matxmat2mat<<<gridMatMul, block_2d>>>(
            d_input_p_T, d_incoming[0], d_gradC[0], cweight_rows, inHeight, cweight_cols
        );
        if (d_input_p_T) CUDA_CHECK(cudaFree(d_input_p_T));

        dim3 gridWeightGrad = calculate_grid_1d(cweight_flat_size, CUDA_BLOCK_SIZE_1D);
        scaleByValue<<<gridWeightGrad, block_1d>>>(d_gradC[0], d_gradC[0], alpha, (int)cweight_flat_size);

        // dL/dB_1: (W x H) * (H x C_1) -> W x C_1
        d_ones = nullptr; d_onesT = nullptr; // Reset for new allocation
        std::vector<float> ones_L0(in_flat_size, 1.0f);
        CUDA_CHECK(cudaMalloc(&d_ones, sizeof(float) * in_flat_size));
        CUDA_CHECK(cudaMemcpy(d_ones, ones_L0.data(), sizeof(float) * in_flat_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_onesT, sizeof(float) * in_p_T_size));
        transpose<<<grid2D_in, block_2d>>>(d_ones, d_onesT, inHeight, inWidth);
        if (d_ones) CUDA_CHECK(cudaFree(d_ones));

        matxmat2mat<<<gridMatMul, block_2d>>>(
            d_onesT, d_incoming[0], d_gradB[0], inWidth, inHeight, firstLayerCols
        );
        if (d_onesT) CUDA_CHECK(cudaFree(d_onesT));

        size_t bweight_flat_size_L0 = bweights[0].size() * bweights[0][0].size();
        scaleByValue<<<calculate_grid_1d(bweight_flat_size_L0, CUDA_BLOCK_SIZE_1D), block_1d>>>(d_gradB[0], d_gradB[0], 1.0f - alpha, (int)bweight_flat_size_L0);

        // --- Update Weights and Copy Results Back ---
        for (int i = 0; i < this->layers; ++i) {
            std::vector<float> flat_c = flatten(cweights[i]);
            size_t c_size = flat_c.size();
            size_t b_size = c_size; // b size must match c size
            dim3 gridUpdate = calculate_grid_1d(c_size, CUDA_BLOCK_SIZE_1D);

            kernelUpdateWeightsElasticNet<<<gridUpdate, block_1d>>>(d_cweights[i], d_gradC[i], (int)c_size,
                                            learningRate, LAMBDA_L1, LAMBDA_L2);
            kernelUpdateWeightsElasticNet<<<gridUpdate, block_1d>>>(d_bweights[i], d_gradB[i], (int)b_size,
                                            learningRate, LAMBDA_L1, LAMBDA_L2);

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

    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn2d::cuBackprop: ") + e.what());
    }
    
    // --- FINAL MEMORY CLEANUP BLOCK ---
    
    if (d_in) CUDA_CHECK(cudaFree(d_in));
    if (d_out) CUDA_CHECK(cudaFree(d_out));
    if (d_exp) CUDA_CHECK(cudaFree(d_exp));
    if (d_err) CUDA_CHECK(cudaFree(d_err));

    // Cleanup of Reusable Buffer d_grad_x_CT
    if (d_grad_x_CT) CUDA_CHECK(cudaFree(d_grad_x_CT));

    // Cleanup layer-specific vectors
    for (int i = 0; i < this->layers; i++) {
        if (d_cweights[i]) CUDA_CHECK(cudaFree(d_cweights[i]));
        if (d_bweights[i]) CUDA_CHECK(cudaFree(d_bweights[i]));
        if (d_gradC[i]) CUDA_CHECK(cudaFree(d_gradC[i]));
        if (d_gradB[i]) CUDA_CHECK(cudaFree(d_gradB[i]));
        if (d_dotProds[i]) CUDA_CHECK(cudaFree(d_dotProds[i]));
        if (d_activate[i]) CUDA_CHECK(cudaFree(d_activate[i]));
        if (d_incoming[i]) CUDA_CHECK(cudaFree(d_incoming[i]));
        
        // d_outgoing, d_dpow, d_dact vectors only have layers-1 elements
        if (i < this->layers - 1) {
            if (d_outgoing[i]) CUDA_CHECK(cudaFree(d_outgoing[i]));
            if (d_dpow[i]) CUDA_CHECK(cudaFree(d_dpow[i]));
            if (d_dact[i]) CUDA_CHECK(cudaFree(d_dact[i]));
        }
    }
    // No other global temporaries need explicit cleanup here as they are reset/freed in their respective blocks.
}

#endif