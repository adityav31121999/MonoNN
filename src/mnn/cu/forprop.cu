#ifdef USE_CU
#include "mnn.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm> // For std::max

/**
 * @brief forprop for mnn using CUDA
 * @param input input vector
 */
void mnn::cuForprop(const std::vector<float>& input)
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
        CUDA_CHECK(cudaMalloc(&d_input, input_size_bytes));
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), input_size_bytes, cudaMemcpyHostToDevice));
        for(int i = 0; i < this->layers; i++) {
            // Flatten host weights/biases for H2D transfer
            std::vector<float> flat_c = flatten(cweights[i]);
            std::vector<float> flat_b = flatten(bweights[i]);
            size_t cweight_size_bytes = flat_c.size() * sizeof(float);
            size_t bweight_size_bytes = flat_b.size() * sizeof(float);
            size_t dotprod_size = dotProds[i].size();
            size_t dotprod_size_bytes = dotprod_size * sizeof(float);

            CUDA_CHECK(cudaMalloc(&d_clayers[i], cweight_size_bytes));
            CUDA_CHECK(cudaMalloc(&d_blayers[i], bweight_size_bytes));
            CUDA_CHECK(cudaMemcpy(d_clayers[i], flat_c.data(), cweight_size_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_blayers[i], flat_b.data(), bweight_size_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_dotProds[i], dotprod_size_bytes));
            CUDA_CHECK(cudaMalloc(&d_activate[i], dotprod_size_bytes));
        }

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

            CUDA_CHECK(cudaMemcpy(dotProds[i].data(), d_dotProds[i], current_out_size * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(activate[i].data(), d_activate[i], current_out_size * sizeof(float), cudaMemcpyDeviceToHost));
        }

        int final_output_size = (int)output.size();
        CUDA_CHECK(cudaMemcpy(output.data(), d_activate[layers - 1], final_output_size * sizeof(float), cudaMemcpyDeviceToHost));

        // free memory
        CUDA_CHECK(cudaFree(d_input));
        for (int i = 0; i < this->layers; i++) {
            CUDA_CHECK(cudaFree(d_clayers[i]));
            CUDA_CHECK(cudaFree(d_blayers[i]));
            CUDA_CHECK(cudaFree(d_dotProds[i]));
            CUDA_CHECK(cudaFree(d_activate[i]));
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
void mnn2d::cuForprop(const std::vector<std::vector<float>>& input)
{
    try {
        float* d_input = nullptr;
        float* d_output = nullptr;
        std::vector<float*> d_clayers(this->layers, nullptr);
        std::vector<float*> d_blayers(this->layers, nullptr);
        std::vector<float*> d_dotProds(this->layers, nullptr);
        std::vector<float*> d_activate(this->layers, nullptr);
        float* current_input_ptr;

        int currentInHeight = input.size();
        int currentInWidth = input[0].size();
        std::vector<float> flat_input = flatten(input);
        size_t input_size_bytes = flat_input.size() * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_input, input_size_bytes));
        CUDA_CHECK(cudaMemcpy(d_input, flat_input.data(), input_size_bytes, cudaMemcpyHostToDevice));

        for(int i = 0; i < this->layers; i++) {
            std::vector<float> flat_c = flatten(cweights[i]);
            std::vector<float> flat_b = flatten(bweights[i]);
            size_t cweight_size_bytes = flat_c.size() * sizeof(float);
            size_t bweight_size_bytes = flat_b.size() * sizeof(float);
            size_t dotprod_size = dotProds[i].size() * dotProds[i][0].size();
            size_t dotprod_size_bytes = dotprod_size * sizeof(float);

            CUDA_CHECK(cudaMalloc(&d_clayers[i], cweight_size_bytes));
            CUDA_CHECK(cudaMalloc(&d_blayers[i], bweight_size_bytes));
            CUDA_CHECK(cudaMemcpy(d_clayers[i], flat_c.data(), cweight_size_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_blayers[i], flat_b.data(), bweight_size_bytes, cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMalloc(&d_dotProds[i], dotprod_size_bytes));
            CUDA_CHECK(cudaMalloc(&d_activate[i], dotprod_size_bytes));
        }

        for(int i = 0; i < this->layers; i++) {
            if (i == 0) {
                current_input_ptr = d_input;
                currentInHeight = input.size();
                currentInWidth = input[0].size();
            }
            else {
                current_input_ptr = d_activate[i-1];
                currentInHeight = activate[i-1].size();
                currentInWidth = activate[i-1][0].size();
            }

            int currentOutWidth = cweights[i][0].size();
            dim3 block_2d(WORKSIZE_2D_X, WORKSIZE_2D_Y);
            dim3 grid_forward = calculate_grid_2d(currentOutWidth, currentInHeight, WORKSIZE_2D_X, WORKSIZE_2D_Y);
            
            kernelLayerForward4<<<grid_forward, block_2d>>>(
                current_input_ptr,
                d_dotProds[i],
                d_clayers[i],
                d_blayers[i],
                currentInHeight,
                currentInWidth,
                currentOutWidth,
                order
            );

            int dotprod_size = dotProds[i].size() * dotProds[i][0].size();

            if (dotprod_size <= WORKSIZE_1D) {
                // Use the single-block reduction for small inputs
                dim3 block_softmax(dotprod_size);
                dim3 grid_softmax(1);

                softmax<<<grid_softmax, block_softmax>>>(
                    d_dotProds[i],
                    d_activate[i],
                    SOFTMAX_TEMP,
                    dotprod_size
                );
            }
            else {
                // Use the two-stage reduction for large inputs
                dim3 block_reduce(WORKSIZE_1D);
                dim3 grid_reduce = calculate_grid_1d(dotprod_size, WORKSIZE_1D);
                int num_work_groups = grid_reduce.x;
                size_t partial_results_size = num_work_groups * 2;

                float* d_partial_results = nullptr;
                CUDA_CHECK(cudaMalloc(&d_partial_results, partial_results_size * sizeof(float)));

                // Launch reduction kernel
                size_t shared_mem_size = WORKSIZE_1D * sizeof(float) * 2; // For local_max and local_sum
                softmax_reduce<<<grid_reduce, block_reduce, shared_mem_size>>>(
                    d_dotProds[i], d_partial_results, dotprod_size, SOFTMAX_TEMP
                );

                // Copy partial results to host to find global max and sum
                std::vector<float> h_partial_results(partial_results_size);
                CUDA_CHECK(cudaMemcpy(h_partial_results.data(), d_partial_results, partial_results_size * sizeof(float), cudaMemcpyDeviceToHost));

                float global_max = -FLT_MAX;
                float global_sum = 0.0f;
                for (int k = 0; k < num_work_groups; ++k) {
                    global_sum += h_partial_results[2 * k];
                    global_max = std::max(global_max, h_partial_results[2 * k + 1]);
                }

                softmax_normalize<<<grid_reduce, block_reduce>>>(
                    d_dotProds[i], d_activate[i], dotprod_size, SOFTMAX_TEMP, global_max, global_sum
                );

                CUDA_CHECK(cudaFree(d_partial_results));
            }

            size_t activate_size_bytes = dotprod_size * sizeof(float);
            CUDA_CHECK(cudaMemcpy(flatten(dotProds[i]).data(), d_dotProds[i], activate_size_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(flatten(activate[i]).data(), d_activate[i], activate_size_bytes, cudaMemcpyDeviceToHost));
        }

        // --- Final Layer Mean Pool ---
        int last_layer_rows = activate[layers - 1].size();
        int last_layer_cols = activate[layers - 1][0].size();
        size_t output_size_bytes = last_layer_cols * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&d_output, output_size_bytes));
        
        dim3 block_pool(WORKSIZE_1D);
        dim3 grid_pool = calculate_grid_1d(last_layer_cols, WORKSIZE_1D);

        meanPool<<<grid_pool, block_pool>>>(
            d_activate[layers - 1],
            d_output,
            last_layer_rows,
            last_layer_cols,
            1 // poolSize is 1 for meanPool over columns (rows are pooled)
        );

        // --- D2H: Final Output ---
        CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_size_bytes, cudaMemcpyDeviceToHost));

        // free memory
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        for (int i = 0; i < this->layers; i++) {
            CUDA_CHECK(cudaFree(d_clayers[i]));
            CUDA_CHECK(cudaFree(d_blayers[i]));
            CUDA_CHECK(cudaFree(d_dotProds[i]));
            CUDA_CHECK(cudaFree(d_activate[i]));
        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn2d::cuForprop: ") + e.what());
    }
}

#endif // USE_CU