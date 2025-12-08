#ifdef USE_CPU
#include "mnn.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>


/**
 * @brief monomial operation for single layer in forprop
 * @param [in] input vector input
 * @param [out] output vector output
 * @param [in] cweights coefficient weights
 * @param [in] bweights bias weights
 * @param [in] n order of monomial
 */
void layerForward(const std::vector<float>& input, std::vector<float>& output, const std::vector<std::vector<float>>& cweights,
                const std::vector<std::vector<float>>& bweights, float n)
{
    if(input.size() != cweights.size()) {
        throw std::runtime_error("input size and cweights rows mismatch: " + std::to_string(input.size()) + " != " + std::to_string(cweights.size()));
    }
    if(input.size() != bweights.size()) {
        throw std::runtime_error("input size and bweights rows mismatch: " + std::to_string(input.size()) + " != " + std::to_string(bweights.size()));
    }
    if(output.size() != cweights[0].size()) {
        throw std::runtime_error("output size and cweights columns mismatch: " + std::to_string(output.size()) + " != " + std::to_string(cweights[0].size()));
    }
    if(output.size() != bweights[0].size()) {
        throw std::runtime_error("output size and bweights columns mismatch: " + std::to_string(output.size()) + " != " + std::to_string(bweights[0].size()));
    }
    std::vector<float> powerIn = power(input, n);
    for(int j = 0; j < cweights[0].size(); j++) {
        float sum = 0.0f;
        for(int i = 0; i < cweights.size(); i++) {
            sum += (powerIn[i] * cweights[i][j]) + bweights[i][j];
        }
        output[j] = sum;
    }
    // std::transform(output.begin(), output.end(), output.begin(), [](float val) { return clamp(val); }); // Removed clamp
}


/**
 * @brief monomial operation for single layer in forprop
 * @param [in] input matrix input
 * @param [out] output matrix output
 * @param [in] cweights coefficient weights
 * @param [in] bweights bias weights
 * @param [in] n order of monomial
 */
void layerForward(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output,
                    const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n)
{
    if (input.empty() || cweights.empty() || bweights.empty()) {
        throw std::invalid_argument("Input and weight matrices cannot be empty.");
    }
    if (input[0].size() != cweights.size()) {
        throw std::runtime_error("Input columns and cweights rows mismatch.");
    }
    if (cweights.size() != bweights.size() || cweights[0].size() != bweights[0].size()) {
        throw std::runtime_error("cweights and bweights dimensions must match.");
    }
    if (output.size() != input.size() || output[0].size() != cweights[0].size()) {
        throw std::runtime_error("Output matrix has incorrect dimensions.");
    }

    // output = (input^n) * cweights + bweights
    std::vector<std::vector<float>> powerIn(input.size(), std::vector<float>(input[0].size(), 0.0f));
    for (size_t i = 0; i < input.size(); ++i) {
        powerIn[i] = power(input[i], n);
    }
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < cweights[0].size(); ++j) {
            float dotProd_ij = 0.0f;
            for (size_t k = 0; k < cweights.size(); ++k) {
                dotProd_ij += (powerIn[i][k] * cweights[k][j]) + bweights[k][j];
            }
            output[i][j] = dotProd_ij; // Assign the final sum, removed clamp
        }
    }
}

//// Batch Forprop Variants ////

/**
 * @brief batch layer forward for mnn with power
 * @param [in] input batch of input vectors
 * @param [out] output batch of output vectors
 * @param [in] cweights coefficient weights
 * @param [in] bweights bias weights
 * @param [in] n order of monomial
 */
void layerForwardBatch(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output,
                       const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n)
{
    if (input.empty()) 
        throw std::runtime_error("Input batch is empty.");
    if (input.size() != output.size()) 
        throw std::runtime_error("Input batch size and output batch size mismatch.");
    if (input[0].size() != cweights.size()) {
        throw std::runtime_error("input size and cweights rows mismatch :)");
    }
    if (input[0].size() != bweights.size()) {
        throw std::runtime_error("input size and bweights rows mismatch :)");
    }
    if (output[0].size() != cweights[0].size()) {
        throw std::runtime_error("output size and cweights columns mismatch :)");
    }
    if (output[0].size() != bweights[0].size()) {
        throw std::runtime_error("output size and bweights columns mismatch :)");
    }

    int batchSize = input.size();
    int inSize = cweights.size();
    int outSize = cweights[0].size();

    for(int b=0; b<batchSize; ++b) {
        std::vector<float> powerIn = power(input[b], n);

        for(int i=0; i<inSize; ++i) {
            float in_val = powerIn[i];
            for(int j=0; j<outSize; ++j) {
                output[b][j] += in_val * cweights[i][j] + bweights[i][j];
            }
        }

        // std::transform(output[b].begin(), output[b].end(), output[b].begin(), [](float val) { return clamp(val); }); // Removed clamp
    }
}


/**
 * @brief batch layer forward for mnn2d with power
 * @param [in] input batch of input matrices
 * @param [out] output batch of output matrices
 * @param [in] cweights coefficient weights
 * @param [in] bweights bias weights
 * @param [in] n order of monomial
 */
void layerForwardBatch(const std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<std::vector<float>>>& output,
                       const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n)
{
    if (input.empty()) return;
    int batchSize = input.size();
    int inHeight = input[0].size();
    int inWidth = input[0][0].size();
    int outWidth = cweights[0].size();

    for(int b=0; b<batchSize; ++b) {
        std::vector<std::vector<float>> powerIn = power(input[b], n);

        for(int r=0; r<inHeight; ++r) {
            for(int k=0; k<inWidth; ++k) {
                float in_val = powerIn[r][k];
                for(int c=0; c<outWidth; ++c) {
                    output[b][r][c] += (in_val * cweights[k][c]) + bweights[k][c];
                }
            }

            // std::transform(output[b][r].begin(), output[b][r].end(), output[b][r].begin(), [](float val) { return clamp(val); }); // Removed clamp
        }
    }
}

// thread versions
#include <thread>
#include <mutex>

/**
 * @brief Threaded monomial operation for single layer in forprop (Vector Input)
 * Parallelizes the output columns.
 */
void layerForwardThread(const std::vector<float>& input, std::vector<float>& output, 
                        const std::vector<std::vector<float>>& cweights,
                        const std::vector<std::vector<float>>& bweights, float n)
{
    // 1. Validation
    if(input.size() != cweights.size()) throw std::runtime_error("input size and cweights rows mismatch");
    if(input.size() != bweights.size()) throw std::runtime_error("input size and bweights rows mismatch");
    if(output.size() != cweights[0].size()) throw std::runtime_error("output size and cweights columns mismatch");
    if(output.size() != bweights[0].size()) throw std::runtime_error("output size and bweights columns mismatch");

    // 2. Pre-calculate power of input (Vector operation is fast enough to do on main thread or duplicate)
    // Doing it once here prevents re-calculation in every thread.
    std::vector<float> powerIn = power(input, n);

    size_t in_size = cweights.size();
    size_t out_cols = cweights[0].size();

    // 3. Thread Setup
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = num_threads == 0 ? 2 : std::min(num_threads, (unsigned int)out_cols);
    std::vector<std::thread> threads;
    size_t chunk_size = out_cols / num_threads;

    // 4. Worker Lambda (Process range of Output Columns)
    auto worker = [&](size_t start_col, size_t end_col) {
        for(size_t j = start_col; j < end_col; j++) {
            float sum = 0.0f;
            for(size_t i = 0; i < in_size; i++) {
                // Determine contribution: (x^n * w) + b
                sum += (powerIn[i] * cweights[i][j]) + bweights[i][j];
            }
            output[j] = sum;
        }
    };

    // 5. Launch and Join
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? out_cols : start + chunk_size;
        threads.emplace_back(worker, start, end);
    }

    for (auto& t : threads) {
        if(t.joinable()) t.join();
    }
}

/**
 * @brief Threaded monomial operation for single layer in forprop (Matrix Input)
 * Parallelizes the rows of the input matrix.
 */
void layerForwardThread(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output, 
                        const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n)
{
    // 1. Validation
    if (input.empty() || cweights.empty() || bweights.empty()) throw std::invalid_argument("Matrices cannot be empty.");
    if (input[0].size() != cweights.size()) throw std::runtime_error("Dimension mismatch input/weights");
    if (output.size() != input.size() || output[0].size() != cweights[0].size()) throw std::runtime_error("Output dimension mismatch");

    size_t num_rows = input.size();
    size_t num_cols_w = cweights[0].size();
    size_t common_dim = cweights.size();

    // 2. Thread Setup
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = num_threads == 0 ? 2 : std::min(num_threads, (unsigned int)num_rows);
    std::vector<std::thread> threads;
    size_t chunk_size = num_rows / num_threads;

    // 3. Worker Lambda (Process range of Input Rows)
    auto worker = [&](size_t start_row, size_t end_row) {
        for(size_t i = start_row; i < end_row; i++) {
            // Calculate power vector for this specific row locally within thread
            std::vector<float> localPowerIn = power(input[i], n);
            
            for(size_t j = 0; j < num_cols_w; j++) {
                float sum = 0.0f;
                for(size_t k = 0; k < common_dim; k++) {
                    sum += (localPowerIn[k] * cweights[k][j]) + bweights[k][j];
                }
                output[i][j] = sum;
            }
        }
    };

    // 4. Launch and Join
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? num_rows : start + chunk_size;
        threads.emplace_back(worker, start, end);
    }

    for (auto& t : threads) {
        if(t.joinable()) t.join();
    }
}

/**
 * @brief Threaded batch layer forward for mnn with power (Batch of Vectors)
 * Parallelizes the Batch dimension.
 */
void layerForwardBatchThread(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output,
                       const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n)
{
    // 1. Validation
    if (input.empty()) throw std::runtime_error("Input batch is empty.");
    if (input.size() != output.size()) throw std::runtime_error("Batch size mismatch.");
    if (input[0].size() != cweights.size()) throw std::runtime_error("Dimension mismatch.");

    size_t batchSize = input.size();
    size_t inSize = cweights.size();
    size_t outSize = cweights[0].size();

    // 2. Thread Setup
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = num_threads == 0 ? 2 : std::min(num_threads, (unsigned int)batchSize);
    std::vector<std::thread> threads;
    size_t chunk_size = batchSize / num_threads;

    // 3. Worker Lambda (Process range of Batch Indices)
    auto worker = [&](size_t start_b, size_t end_b) {
        for(size_t b = start_b; b < end_b; ++b) {
            std::vector<float> powerIn = power(input[b], n);

            // Optimized loop order: Outer 'j', Inner 'i'. 
            // The original sequential code loops i then j and does `output[b][j] += ...`
            // We compute the full delta for output[b][j] locally to avoid repeated memory writes.
            for(size_t j = 0; j < outSize; ++j) {
                float partial_sum = 0.0f;
                for(size_t i = 0; i < inSize; ++i) {
                    partial_sum += powerIn[i] * cweights[i][j] + bweights[i][j];
                }
                // Accumulate into existing output (matching behavior of sequential version)
                output[b][j] += partial_sum;
            }
        }
    };

    // 4. Launch and Join
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? batchSize : start + chunk_size;
        threads.emplace_back(worker, start, end);
    }

    for (auto& t : threads) {
        if(t.joinable()) t.join();
    }
}

/**
 * @brief Threaded batch layer forward for mnn2d with power (Batch of Matrices)
 * Parallelizes the Batch dimension.
 */
void layerForwardBatchThread(const std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<std::vector<float>>>& output,
                       const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights, float n)
{
    if (input.empty()) return;
    
    size_t batchSize = input.size();
    size_t inHeight = input[0].size();
    size_t inWidth = input[0][0].size();
    size_t outWidth = cweights[0].size();

    // 1. Thread Setup
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = num_threads == 0 ? 2 : std::min(num_threads, (unsigned int)batchSize);
    std::vector<std::thread> threads;
    size_t chunk_size = batchSize / num_threads;

    // 2. Worker Lambda
    auto worker = [&](size_t start_b, size_t end_b) {
        for(size_t b = start_b; b < end_b; ++b) {
            std::vector<std::vector<float>> powerIn = power(input[b], n);

            for(size_t r = 0; r < inHeight; ++r) {
                // Optimization: Loop c (output col) then k (input col) 
                // to use local register for accumulation before writing to memory
                for(size_t c = 0; c < outWidth; ++c) {
                    float sum = 0.0f;
                    for(size_t k = 0; k < inWidth; ++k) {
                        float in_val = powerIn[r][k];
                        sum += (in_val * cweights[k][c]) + bweights[k][c];
                    }
                    output[b][r][c] += sum;
                }
            }
        }
    };

    // 3. Launch and Join
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? batchSize : start + chunk_size;
        threads.emplace_back(worker, start, end);
    }

    for (auto& t : threads) {
        if(t.joinable()) t.join();
    }
}

#endif