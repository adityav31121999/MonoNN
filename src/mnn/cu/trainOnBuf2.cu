#ifdef USE_CU
#include "mnn.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>

/**
 * @brief Trains the mnn network on a batch of data with optimized buffer management.
 * @param inputs A vector of input vectors.
 * @param targets A vector of target vectors.
 */
void mnn::cuBufTrainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets) {
    if (inputs.size() != targets.size() || inputs.empty()) {
        throw std::invalid_argument("Invalid batch training data.");
    }
    this->batchSize = inputs.size();

    // --- Buffer Allocation ---
    float *d_input_batch = nullptr, *d_target_batch = nullptr;
    std::vector<float*> d_cweights(layers), d_bweights(layers), d_gradC(layers), d_gradB(layers);
    std::vector<float*> d_dotProds(layers), d_activate(layers);
    float *d_totalCgrad = nullptr, *d_totalBgrad = nullptr;

    try {
        // Flatten inputs and targets for single transfers
        std::vector<float> flat_inputs, flat_targets;
        for(const auto& vec : inputs) flat_inputs.insert(flat_inputs.end(), vec.begin(), vec.end());
        for(const auto& vec : targets) flat_targets.insert(flat_targets.end(), vec.begin(), vec.end());

        CU_CHECK(cudaMalloc(&d_input_batch, flat_inputs.size() * sizeof(float)));
        CU_CHECK(cudaMemcpy(d_input_batch, flat_inputs.data(), flat_inputs.size() * sizeof(float), cudaMemcpyHostToDevice));
        CU_CHECK(cudaMalloc(&d_target_batch, flat_targets.size() * sizeof(float)));
        CU_CHECK(cudaMemcpy(d_target_batch, flat_targets.data(), flat_targets.size() * sizeof(float), cudaMemcpyHostToDevice));

        size_t max_total_grad_size = 0;
        for (int i = 0; i < layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            size_t layer_output_size = batchSize * width[i];
            max_total_grad_size = std::max(max_total_grad_size, c_size);

            CU_CHECK(cudaMalloc(&d_cweights[i], c_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_bweights[i], b_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_gradC[i], c_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_gradB[i], b_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_dotProds[i], layer_output_size * sizeof(float)));
            CU_CHECK(cudaMalloc(&d_activate[i], layer_output_size * sizeof(float)));
        }
        CU_CHECK(cudaMalloc(&d_totalCgrad, max_total_grad_size * batchSize * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_totalBgrad, max_total_grad_size * batchSize * sizeof(float)));

        int totalEpochs = 0;
        if (this->epochs < 1) this->epochs = 100;
        float initialLR = this->learningRate;

        // --- Training Loop ---
        while (true) {
            totalEpochs++;

            for (int i = 0; i < layers; ++i) {
                std::vector<float> flat_c = flatten(cweights[i]);
                std::vector<float> flat_b = flatten(bweights[i]);
                CU_CHECK(cudaMemcpy(d_cweights[i], flat_c.data(), flat_c.size() * sizeof(float), cudaMemcpyHostToDevice));
                CU_CHECK(cudaMemcpy(d_bweights[i], flat_b.data(), flat_b.size() * sizeof(float), cudaMemcpyHostToDevice));
            }

            // --- Forward Propagation ---
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
                std::cout << "All " << inputs.size() << " outputs correct after " << totalEpochs << " epochs. Loss: " << currloss << std::endl;
                break;
            } else {
                std::cout << "-> Epoch: " << totalEpochs << "\tPredictions: " << correct_predictions << "/" << inputs.size()
                          << "\tAvg. CE Loss: " << currloss << std::endl;
            }

            if (totalEpochs >= this->epochs) {
                std::cout << "Epoch limit reached. Continuing training." << std::endl;
                this->epochs += 50;
            }

            // --- Backward Propagation ---
            cuBackprop(targets);

            for (int i = 0; i < layers; ++i) {
                size_t c_size = cweights[i].size() * cweights[i][0].size();
                size_t b_size = bweights[i].size() * bweights[i][0].size();
                std::vector<float> c_upd(c_size), b_upd(b_size);
                CU_CHECK(cudaMemcpy(c_upd.data(), d_cweights[i], c_size * sizeof(float), cudaMemcpyDeviceToHost));
                CU_CHECK(cudaMemcpy(b_upd.data(), d_bweights[i], b_size * sizeof(float), cudaMemcpyDeviceToHost));
                cweights[i] = reshape(c_upd, cweights[i].size(), cweights[i][0].size());
                bweights[i] = reshape(b_upd, bweights[i].size(), bweights[i][0].size());
            }
        }
        this->learningRate = initialLR;

    } catch (const std::runtime_error& e) {
        std::cerr << "Error during cuBufTrainBatch (mnn): " << e.what() << std::endl;
    }

    // --- Buffer Cleanup ---
    cudaFree(d_input_batch); cudaFree(d_target_batch);
    cudaFree(d_totalCgrad); cudaFree(d_totalBgrad);
    for (int i = 0; i < layers; ++i) {
        cudaFree(d_cweights[i]); cudaFree(d_bweights[i]);
        cudaFree(d_gradC[i]); cudaFree(d_gradB[i]);
        cudaFree(d_dotProds[i]); cudaFree(d_activate[i]);
    }
}

/**
 * @brief Trains the mnn2d network on a batch of data with optimized buffer management.
 * @param inputs A vector of input matrices.
 * @param targets A vector of target vectors.
 */
void mnn2d::cuBufTrainBatch(const std::vector<std::vector<std::vector<float>>>& inputs, const std::vector<std::vector<float>>& targets) {
    if (inputs.size() != targets.size() || inputs.empty()) {
        throw std::invalid_argument("Invalid batch training data.");
    }
    this->batchSize = inputs.size();

    // --- Buffer Allocation ---
    float *d_input_batch = nullptr, *d_target_batch = nullptr, *d_final_output = nullptr;
    std::vector<float*> d_cweights(layers), d_bweights(layers), d_gradC(layers), d_gradB(layers);
    std::vector<float*> d_dotProds(layers), d_activate(layers);
    float *d_totalCgrad = nullptr, *d_totalBgrad = nullptr;
    float *d_grad_x_CT = nullptr, *d_dprev_p = nullptr, *d_dprev_act = nullptr;

    try {
        std::vector<float> flat_inputs;
        for(const auto& mat : inputs) for(const auto& row : mat) flat_inputs.insert(flat_inputs.end(), row.begin(), row.end());
        std::vector<float> flat_targets;
        for(const auto& vec : targets) flat_targets.insert(flat_targets.end(), vec.begin(), vec.end());

        CU_CHECK(cudaMalloc(&d_input_batch, flat_inputs.size() * sizeof(float)));
        CU_CHECK(cudaMemcpy(d_input_batch, flat_inputs.data(), flat_inputs.size() * sizeof(float), cudaMemcpyHostToDevice));
        CU_CHECK(cudaMalloc(&d_target_batch, flat_targets.size() * sizeof(float)));
        CU_CHECK(cudaMemcpy(d_target_batch, flat_targets.data(), flat_targets.size() * sizeof(float), cudaMemcpyHostToDevice));
        CU_CHECK(cudaMalloc(&d_final_output, batchSize * outWidth * sizeof(float)));

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

        int totalEpochs = 0;
        if (this->epochs < 1) this->epochs = 100;
        float initialLR = this->learningRate;

        // --- Training Loop ---
        while (true) {
            totalEpochs++;

            for (int i = 0; i < layers; ++i) {
                std::vector<float> flat_c = flatten(cweights[i]);
                std::vector<float> flat_b = flatten(bweights[i]);
                CU_CHECK(cudaMemcpy(d_cweights[i], flat_c.data(), flat_c.size() * sizeof(float), cudaMemcpyHostToDevice));
                CU_CHECK(cudaMemcpy(d_bweights[i], flat_b.data(), flat_b.size() * sizeof(float), cudaMemcpyHostToDevice));
            }

            // --- Forward Propagation ---
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
                std::cout << "All " << inputs.size() << " outputs correct after " << totalEpochs << " epochs. Loss: " << currloss << std::endl;
                break;
            } else {
                std::cout << " | Epoch: " << totalEpochs << "\tPredictions: " << correct_predictions << "/" << inputs.size()
                          << "\tAvg. CE Loss: " << currloss << std::endl;
            }

            if (totalEpochs >= this->epochs) {
                std::cout << "Epoch limit reached. Continuing training." << std::endl;
                this->epochs += 10;
            }

            // --- Backward Propagation ---
            cuBackprop(targets);

            for (int i = 0; i < layers; ++i) {
                size_t c_size = cweights[i].size() * cweights[i][0].size();
                size_t b_size = bweights[i].size() * bweights[i][0].size();
                std::vector<float> c_upd(c_size), b_upd(b_size);
                CU_CHECK(cudaMemcpy(c_upd.data(), d_cweights[i], c_size * sizeof(float), cudaMemcpyDeviceToHost));
                CU_CHECK(cudaMemcpy(b_upd.data(), d_bweights[i], b_size * sizeof(float), cudaMemcpyDeviceToHost));
                cweights[i] = reshape(c_upd, cweights[i].size(), cweights[i][0].size());
                bweights[i] = reshape(b_upd, bweights[i].size(), bweights[i][0].size());
            }
        }
        this->learningRate = initialLR;

    } catch (const std::runtime_error& e) {
        std::cerr << "Error during cuBufTrainBatch (mnn2d): " << e.what() << std::endl;
    }

    // --- Buffer Cleanup ---
    cudaFree(d_input_batch); cudaFree(d_target_batch); cudaFree(d_final_output);
    cudaFree(d_totalCgrad); cudaFree(d_totalBgrad);
    cudaFree(d_grad_x_CT); cudaFree(d_dprev_p); cudaFree(d_dprev_act);
    for (int i = 0; i < layers; ++i) {
        cudaFree(d_cweights[i]); cudaFree(d_bweights[i]);
        cudaFree(d_gradC[i]); cudaFree(d_gradB[i]);
        cudaFree(d_dotProds[i]); cudaFree(d_activate[i]);
    }
}

#endif // USE_CU