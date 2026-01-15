#ifdef USE_CU
#include "mnn.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include "operators.hpp"

/**
 * @brief batch forprop for mnn using CUDA
 * @param input input vector
 */
void mnn::cuForprop(const std::vector<std::vector<float>>& input)
{
    try {
        this->batchSize = input.size();
        if (this->batchSize == 0) return;

        // Flatten the batch input for a single transfer
        std::vector<float> flat_input;
        for(const auto& vec : input) {
            flat_input.insert(flat_input.end(), vec.begin(), vec.end());
        }
        size_t single_input_size = input[0].size();
        size_t total_input_size = flat_input.size();

        // device buffers
        float *d_input, *d_current_act;
        std::vector<float*> d_clayers(this->layers);
        std::vector<float*> d_blayers(this->layers);
        std::vector<float*> d_dotProds(this->layers);
        std::vector<float*> d_activate(this->layers);

        // Create and write to input buffer
        CU_CHECK(cudaMalloc((void**)&d_input, sizeof(float) * total_input_size));
        CU_CHECK(cudaMemcpy(d_input, flat_input.data(), sizeof(float) * total_input_size, cudaMemcpyHostToDevice));

        // Create all other buffers and copy weight data
        for(int i = 0; i < this->layers; i++) {
            size_t cweight_size = cweights[i].size() * cweights[i][0].size();
            size_t bweight_size = bweights[i].size() * bweights[i][0].size();
            std::vector<float> flat_cweights = flatten(cweights[i]);
            std::vector<float> flat_bweights = flatten(bweights[i]);

            CU_CHECK(cudaMalloc((void**)&d_clayers[i], sizeof(float) * cweight_size));
            CU_CHECK(cudaMemcpy(d_clayers[i], flat_cweights.data(), sizeof(float) * cweight_size, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMalloc((void**)&d_blayers[i], sizeof(float) * bweight_size));
            CU_CHECK(cudaMemcpy(d_blayers[i], flat_bweights.data(), sizeof(float) * bweight_size, cudaMemcpyHostToDevice));
            
            size_t layer_output_size = batchSize * width[i];
            CU_CHECK(cudaMalloc((void**)&d_dotProds[i], sizeof(float) * layer_output_size));
            CU_CHECK(cudaMalloc((void**)&d_activate[i], sizeof(float) * layer_output_size));
        }

        // first layer forward
        d_current_act = d_input;
        dim3 blockForward(WORKSIZE_1D, 1, 1);
        dim3 gridForward(width[0] / WORKSIZE_1D + 1, batchSize, 1);
        kernelLayerForwardBatch2<<<gridForward, blockForward>>>(d_current_act, d_dotProds[0], d_clayers[0], d_blayers[0], batchSize, single_input_size, width[0], order);
        CU_CHECK(cudaGetLastError());

        // activation
        int size_layer0 = batchSize * width[0];
        dim3 blockSig(WORKSIZE_1D, 1, 1);
        dim3 gridSig((size_layer0 + WORKSIZE_1D - 1) / WORKSIZE_1D, 1, 1);
        sigmoid<<<gridSig, blockSig>>>(d_dotProds[0], d_activate[0], size_layer0);
        CU_CHECK(cudaGetLastError());

        // for second to last layer
        for(int i = 1; i < this->layers; i++) {
            d_current_act = d_activate[i-1];
            // ith layer forward
            gridForward = dim3(width[i] / WORKSIZE_1D + 1, batchSize, 1);
            kernelLayerForwardBatch2<<<gridForward, blockForward>>>(d_current_act, d_dotProds[i], d_clayers[i], d_blayers[i], batchSize, width[i-1], width[i], order);
            CU_CHECK(cudaGetLastError());

            // activate
            int size_layer_i = batchSize * width[i];
            gridSig = dim3((size_layer_i + WORKSIZE_1D - 1) / WORKSIZE_1D, 1, 1);
            sigmoid<<<gridSig, blockSig>>>(d_dotProds[i], d_activate[i], size_layer_i);
            CU_CHECK(cudaGetLastError());
        }

        // Read the final activation and other results back to the host
        std::vector<float> final_activations(batchSize * outSize);
        CU_CHECK(cudaMemcpy(final_activations.data(), d_activate[layers - 1], sizeof(float) * final_activations.size(), cudaMemcpyDeviceToHost));

        for(int i = 0; i < batchSize; ++i) {
            std::copy(final_activations.begin() + i * outSize, final_activations.begin() + (i + 1) * outSize, outputBatch[i].begin());
        }

        // copy dot and acts
        for(int i=0; i<layers; ++i) {
            std::vector<float> dot_flat(batchSize * width[i]);
            std::vector<float> act_flat(batchSize * width[i]);
            CU_CHECK(cudaMemcpy(dot_flat.data(), d_dotProds[i], sizeof(float) * dot_flat.size(), cudaMemcpyDeviceToHost));
            CU_CHECK(cudaMemcpy(act_flat.data(), d_activate[i], sizeof(float) * act_flat.size(), cudaMemcpyDeviceToHost));
            for(int j=0; j<batchSize; ++j) {
                std::copy(dot_flat.begin() + j * width[i], dot_flat.begin() + (j+1) * width[i], dotBatch[i][j].begin());
                std::copy(act_flat.begin() + j * width[i], act_flat.begin() + (j+1) * width[i], actBatch[i][j].begin());
            }
        }
        outputBatch = actBatch[layers-1];
        for (int i = 0; i < batchSize; ++i) {
            outputBatch[i] = softmax(outputBatch[i], SOFTMAX_TEMP);
        }

        // Free device memory
        cudaFree(d_input);
        for(int i = 0; i < this->layers; i++) {
            cudaFree(d_clayers[i]);
            cudaFree(d_blayers[i]);
            cudaFree(d_dotProds[i]);
            cudaFree(d_activate[i]);
        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn::cuForprop: ") + e.what());
    }
}


/**
 * @brief forprop for mnn2d using CUDA
 * @param input input matrix
 */
void mnn2d::cuForprop(const std::vector<std::vector<std::vector<float>>>& input)
{
    try {
        this->batchSize = input.size();
        if (this->batchSize == 0) return;

        std::vector<float> flat_input;
        for(const auto& mat : input) {
            for(const auto& row : mat) {
                flat_input.insert(flat_input.end(), row.begin(), row.end());
            }
        }
        size_t single_input_height = input[0].size();
        size_t single_input_width = input[0][0].size();
        size_t total_input_size = flat_input.size();

        // device buffers
        float *d_input, *d_current_act;
        std::vector<float*> d_clayers(this->layers);
        std::vector<float*> d_blayers(this->layers);
        std::vector<float*> d_dotProds(this->layers);
        std::vector<float*> d_activate(this->layers);

        // Create and write to input buffer
        CU_CHECK(cudaMalloc((void**)&d_input, sizeof(float) * total_input_size));
        CU_CHECK(cudaMemcpy(d_input, flat_input.data(), sizeof(float) * total_input_size, cudaMemcpyHostToDevice));

        for(int i = 0; i < this->layers; i++) {
            size_t cweight_size = cweights[i].size() * cweights[i][0].size();
            size_t bweight_size = bweights[i].size() * bweights[i][0].size();
            std::vector<float> flat_cweights = flatten(cweights[i]);
            std::vector<float> flat_bweights = flatten(bweights[i]);

            CU_CHECK(cudaMalloc((void**)&d_clayers[i], sizeof(float) * cweight_size));
            CU_CHECK(cudaMemcpy(d_clayers[i], flat_cweights.data(), sizeof(float) * cweight_size, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMalloc((void**)&d_blayers[i], sizeof(float) * bweight_size));
            CU_CHECK(cudaMemcpy(d_blayers[i], flat_bweights.data(), sizeof(float) * bweight_size, cudaMemcpyHostToDevice));
            
            size_t layer_output_size = batchSize * inHeight * width[i];
            CU_CHECK(cudaMalloc((void**)&d_dotProds[i], sizeof(float) * layer_output_size));
            CU_CHECK(cudaMalloc((void**)&d_activate[i], sizeof(float) * layer_output_size));
        }

        // First layer
        d_current_act = d_input;
        dim3 blockForward(WORKSIZE_2D_X, WORKSIZE_2D_Y, 1);
        dim3 gridForward(width[0] / WORKSIZE_2D_X + 1, single_input_height / WORKSIZE_2D_Y + 1, batchSize);
        kernelLayerForwardBatch4<<<gridForward, blockForward>>>(d_current_act, d_dotProds[0], d_clayers[0], d_blayers[0], batchSize, single_input_height, single_input_width, width[0], order);
        CU_CHECK(cudaGetLastError());

        // Activation for first layer
        size_t dotprod_size_layer0 = batchSize * inHeight * width[0];
        dim3 blockSoftmax(WORKSIZE_1D, 1, 1);
        dim3 gridRelu((dotprod_size_layer0 + WORKSIZE_1D - 1) / WORKSIZE_1D, 1, 1);
        relu<<<gridRelu, blockSoftmax>>>(d_dotProds[0], d_activate[0], dotprod_size_layer0);
        CU_CHECK(cudaGetLastError());

        // hidden layers
        for(int i = 1; i < this->layers; i++) {
            d_current_act = d_activate[i-1];
            gridForward = dim3(width[i] / WORKSIZE_2D_X + 1, inHeight / WORKSIZE_2D_Y + 1, batchSize);
            kernelLayerForwardBatch4<<<gridForward, blockForward>>>(d_current_act, d_dotProds[i], d_clayers[i], d_blayers[i], batchSize, inHeight, width[i-1], width[i], order);
            CU_CHECK(cudaGetLastError());

            // softmax to the flattened dot product
            size_t dotprod_size_layer_i = batchSize * inHeight * width[i];
            gridRelu = dim3((dotprod_size_layer_i + WORKSIZE_1D - 1) / WORKSIZE_1D, 1, 1);
            relu<<<gridRelu, blockSoftmax>>>(d_dotProds[i], d_activate[i], dotprod_size_layer_i);
            CU_CHECK(cudaGetLastError());
        }

        // Mean pool the final activation layer
        float* d_final_output;
        CU_CHECK(cudaMalloc((void**)&d_final_output, sizeof(float) * batchSize * outSize));
        dim3 blockPool(WORKSIZE_1D, 1, 1);
        dim3 gridPool(outSize / WORKSIZE_1D + 1, batchSize, 1);
        meanPool<<<gridPool, blockPool>>>(
            d_activate[layers-1],
            d_final_output,
            inHeight,
            outSize,
            batchSize
        );
        CU_CHECK(cudaGetLastError());

        // Read final output and intermediate results back to host
        std::vector<float> final_output_flat(batchSize * outSize);
        CU_CHECK(cudaMemcpy(final_output_flat.data(), d_final_output, sizeof(float) * final_output_flat.size(), cudaMemcpyDeviceToHost));
        for(int i=0; i<batchSize; ++i) {
            std::copy(final_output_flat.begin() + i * outSize, final_output_flat.begin() + (i+1) * outSize, outputBatch[i].begin());
        }
        outputBatch = reshape(final_output_flat, batchSize, outSize);

        // copy dot and acts
        for(int i=0; i<layers; ++i) {
            size_t layer_size = batchSize * inHeight * width[i];
            std::vector<float> dot_flat(layer_size);
            std::vector<float> act_flat(layer_size);
            CU_CHECK(cudaMemcpy(dot_flat.data(), d_dotProds[i], sizeof(float) * layer_size, cudaMemcpyDeviceToHost));
            CU_CHECK(cudaMemcpy(act_flat.data(), d_activate[i], sizeof(float) * layer_size, cudaMemcpyDeviceToHost));

            for(int j=0; j<batchSize; ++j) {
                size_t single_item_size = inHeight * width[i];
                std::vector<float> single_dot(dot_flat.begin() + j * single_item_size, dot_flat.begin() + (j+1) * single_item_size);
                std::vector<float> single_act(act_flat.begin() + j * single_item_size, act_flat.begin() + (j+1) * single_item_size);
                dotBatch[i][j] = reshape(single_dot, inHeight, width[i]);
                actBatch[i][j] = reshape(single_act, inHeight, width[i]);
            }
        }

        for(int i=0; i<batchSize; ++i) {
            outputBatch[i] = softmax(outputBatch[i], SOFTMAX_TEMP);
        }

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_final_output);
        for(int i = 0; i < this->layers; i++) {
            cudaFree(d_clayers[i]);
            cudaFree(d_blayers[i]);
            cudaFree(d_dotProds[i]);
            cudaFree(d_activate[i]);
        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn2d::cuForprop: ") + e.what());
    }
}

#endif // USE_CU