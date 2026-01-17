#ifdef USE_CU
#include "mnn1d.hpp"

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm> // For std::max

/**
 * @brief forprop for mnn1d using CUDA
 * @param input input vector
 */
void mnn1d::cuForprop(const std::vector<float>& input)
{
    try {
        // --- Device Memory Pointers ---
        float* d_input = nullptr;
        std::vector<float*> d_clayers(this->layers, nullptr);
        std::vector<float*> d_blayers(this->layers, nullptr);
        std::vector<float*> d_dotProds(this->layers, nullptr);
        std::vector<float*> d_activate(this->layers, nullptr);
        float* current_input_ptr; // Pointer for the input to the current layer

        size_t input_size_bytes = input.size() * sizeof(float);
        CU_CHECK(cudaMalloc(&d_input, input_size_bytes));
        CU_CHECK(cudaMemcpy(d_input, input.data(), input_size_bytes, cudaMemcpyHostToDevice));
        for(int i = 0; i < this->layers; i++) {
            // Flatten host weights/biases for H2D transfer
            std::vector<float> flat_c = flatten(cweights[i]);
            std::vector<float> flat_b = flatten(bweights[i]);
            size_t cweight_size_bytes = flat_c.size() * sizeof(float);
            size_t bweight_size_bytes = flat_b.size() * sizeof(float);
            size_t dotprod_size = dotProds[i].size();
            size_t dotprod_size_bytes = dotprod_size * sizeof(float);

            CU_CHECK(cudaMalloc(&d_clayers[i], cweight_size_bytes));
            CU_CHECK(cudaMalloc(&d_blayers[i], bweight_size_bytes));
            CU_CHECK(cudaMemcpy(d_clayers[i], flat_c.data(), cweight_size_bytes, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMemcpy(d_blayers[i], flat_b.data(), bweight_size_bytes, cudaMemcpyHostToDevice));
            CU_CHECK(cudaMalloc(&d_dotProds[i], dotprod_size_bytes));
            CU_CHECK(cudaMalloc(&d_activate[i], dotprod_size_bytes));
        }

        // forprop on layers
        for(int i = 0; i < this->layers; i++) {
            if (i == 0) {
                current_input_ptr = d_input;
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
                d_clayers[i],
                d_blayers[i],
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

            CU_CHECK(cudaMemcpy(dotProds[i].data(), d_dotProds[i], current_out_size * sizeof(float), cudaMemcpyDeviceToHost));
            CU_CHECK(cudaMemcpy(activate[i].data(), d_activate[i], current_out_size * sizeof(float), cudaMemcpyDeviceToHost));
        }

        int final_output_size = (int)output.size();
        CU_CHECK(cudaMemcpy(output.data(), d_activate[layers - 1], final_output_size * sizeof(float), cudaMemcpyDeviceToHost));
        output = softmax(output, SOFTMAX_TEMP);

        // free memory
        CU_CHECK(cudaFree(d_input));
        for (int i = 0; i < this->layers; i++) {
            CU_CHECK(cudaFree(d_clayers[i]));
            CU_CHECK(cudaFree(d_blayers[i]));
            CU_CHECK(cudaFree(d_dotProds[i]));
            CU_CHECK(cudaFree(d_activate[i]));
        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn1d::cuForprop: ") + e.what());
    }
}

#endif // USE_CU