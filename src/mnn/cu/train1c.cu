#ifdef USE_CU
#include "mnn.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm> // For std::max
#include <cmath>     // For std::ceil
#include <limits>    // For std::numeric_limits

/**
 * @brief trains the mnn network on a single input-target pair for 1 cycle using OpenCL.
 * @param input The input vector.
 * @param target The target output vector.
 * @param useBuffer 0 for stand alone functions or 1 for all-buffers-in-single function 
 */
void mnn::cuTrain1c(const std::vector<float>& input, const std::vector<float>& target, bool useBuffer) {
    if (useBuffer == 0) {
        // 1. Forward propagation
        this->input = softmax(input);
        cuForprop(this->input);

        if(maxIndex(output) == maxIndex(target)) {
            std::cout << "Correct output predicted with loss " << crossEntropy(output, target) << "." << std::endl;
        }
        else {
            // check for error and break if acceptable
            currloss = crossEntropy(output, target);
            std::cout << "Current CE Loss: " << currloss << std::endl;

            // 2. Backward propagation
            this->target = target;
            cuBackprop(this->target);
            prevloss = currloss;
        }
    }
    else {
        // 1. Forward propagation
    // --- Buffer Allocation ---
        float *d_in = nullptr, *d_exp = nullptr, *d_out = nullptr, *d_err = nullptr, *d_ones = nullptr;
        std::vector<float*> d_cweights(layers), d_bweights(layers), d_gradC(layers), d_gradB(layers);
        std::vector<float*> d_dotProds(layers), d_activate(layers), d_incoming(layers);
        std::vector<float*> d_dpow(layers > 1 ? layers - 1 : 0), d_dact(layers > 1 ? layers - 1 : 0);
        float *d_preoutgoing_l = nullptr, *d_outgoing_l = nullptr;

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
                CU_CHECK(cudaMalloc(&d_outgoing_l, sizeof(float) * max_outgoing_size));
            }
            for (int i = 0; i < layers - 1; ++i) {
                size_t act_size = activate[i].size();
                CU_CHECK(cudaMalloc(&d_dpow[i], act_size * sizeof(float)));
                CU_CHECK(cudaMalloc(&d_dact[i], act_size * sizeof(float)));
            }
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
        cuForprop(input); // This will now use pre-allocated buffers

        CU_CHECK(cudaMemcpy(output.data(), d_activate[layers - 1], output.size() * sizeof(float), cudaMemcpyDeviceToHost));

        if(maxIndex(output) == maxIndex(target)) {
            std::cout << "Correct output predicted with loss " << crossEntropy(output, target) << "." << std::endl;
        }
        else {
            currloss = crossEntropy(output, target);
            std::cout << "Current CE Loss: " << currloss << std::endl;

            // 2. Backward propagation
            this->target = target;
            zeroGradients();

            cuBackprop(target); // This will now use pre-allocated buffers
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
        cudaFree(d_preoutgoing_l); cudaFree(d_outgoing_l);
        for (int i = 0; i < layers; ++i) {
            cudaFree(d_cweights[i]); cudaFree(d_bweights[i]);
            cudaFree(d_gradC[i]); cudaFree(d_gradB[i]);
            cudaFree(d_dotProds[i]); cudaFree(d_activate[i]);
            cudaFree(d_incoming[i]);
            if (i < layers - 1) {
                cudaFree(d_dpow[i]); cudaFree(d_dact[i]);
            }
        }
    }
}

/**
 * @brief trains the mnn2d network on a single input-target pair using OpenCL.
 * @param input The input matrix.
 * @param target The target vector (corresponding to the pooled output).
 * @param useBuffer 0 for stand alone functions or 1 for all-buffers-in-single function 
 */
void mnn2d::cuTrain1c(const std::vector<std::vector<float>>& input, const std::vector<float>& target, bool useBuffer) {
    if (useBuffer == 0) {
        // 1. Forward propagation
        this->input = softmax(input);
        cuForprop(this->input);

        if(maxIndex(output) == maxIndex(target)) {
            std::cout << "Correct output predicted with loss " << crossEntropy(output, target) << "." << std::endl;
        }
        else {
            // check for error and break if acceptable
            currloss = crossEntropy(output, target);
            std::cout << "Current CE Loss: " << currloss << std::endl;

            // 2. Backward propagation
            this->target = target;
            cuBackprop(this->target);
            prevloss = currloss;
        }
    }
    else {
        // --- Buffer Allocation ---
        float *d_in = nullptr, *d_exp = nullptr, *d_out = nullptr, *d_err = nullptr, *d_ones = nullptr;
        std::vector<float*> d_cweights(layers), d_bweights(layers), d_gradC(layers), d_gradB(layers);
        std::vector<float*> d_dotProds(layers), d_activate(layers), d_incoming(layers);
        std::vector<float*> d_dpow(layers > 1 ? layers - 1 : 0), d_dact(layers > 1 ? layers - 1 : 0);
        float *d_preoutgoing_l = nullptr, *d_outgoing_l = nullptr;

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
                CU_CHECK(cudaMalloc(&d_outgoing_l, sizeof(float) * max_outgoing_size));
            }
            for (int i = 0; i < layers - 1; ++i) {
                size_t act_size = activate[i].size();
                CU_CHECK(cudaMalloc(&d_dpow[i], act_size * sizeof(float)));
                CU_CHECK(cudaMalloc(&d_dact[i], act_size * sizeof(float)));
            }
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
        cuForprop(input); // This will now use pre-allocated buffers

        // Copy output D2H to check for correctness and loss
        CU_CHECK(cudaMemcpy(output.data(), d_activate[layers - 1], output.size() * sizeof(float), cudaMemcpyDeviceToHost));

        if(maxIndex(output) == maxIndex(target)) {
            std::cout << "Correct output predicted with loss " << crossEntropy(output, target) << "." << std::endl;
        }
        else {
            currloss = crossEntropy(output, target);
            std::cout << "Current CE Loss: " << currloss << std::endl;

            // 2. Backward propagation
            this->target = target;
            zeroGradients();

            // --- Backward Propagation ---
            cuBackprop(target); // This will now use pre-allocated buffers
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
        cudaFree(d_preoutgoing_l); cudaFree(d_outgoing_l);
        for (int i = 0; i < layers; ++i) {
            cudaFree(d_cweights[i]); cudaFree(d_bweights[i]);
            cudaFree(d_gradC[i]); cudaFree(d_gradB[i]);
            cudaFree(d_dotProds[i]); cudaFree(d_activate[i]);
            cudaFree(d_incoming[i]);
            if (i < layers - 1) {
                cudaFree(d_dpow[i]); cudaFree(d_dact[i]);
            }
        }
    }
}

#endif // USE_CU