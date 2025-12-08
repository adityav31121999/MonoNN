#ifdef USE_CPU
#include "mnn.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>

// BATCH BACKPROP FOR MNN

/**
 * @brief batch layer backprop for mnn for first layer
 * @param[in] incoming batch of incoming gradients (dL/dz_l)
 * @param[in] input input to mnn is used here
 * @param[in] C current layers coefficients weights matrix
 * @param[out] gradc accumulated gradients for C matrix
 * @param[out] gradb accumulated gradients for B matrix
 * @param[in] m order of monomial
 * @param[in] alpha gradient splitting factor
 */
void layerBackwardBatch(const std::vector<std::vector<float>>& incoming,
                        const std::vector<std::vector<float>>& input, 
                        std::vector<std::vector<float>>& C,
                        std::vector<std::vector<float>>& gradc,
                        std::vector<std::vector<float>>& gradb,
                        float m,
                        float alpha)
{
    int batchSize = incoming.size();

    // accumulate gradients
    for(int b = 0; b < batchSize; ++b) {
        std::vector<float> prev_p = power(input[b], m);
        for(int i = 0; i < prev_p.size(); i++) {
            for(int j = 0; j < incoming[b].size(); j++) {
                gradc[i][j] += alpha * prev_p[i] * incoming[b][j];
                gradb[i][j] += (1.0f - alpha) * incoming[b][j];
            }
        }
    }

    // Average gradients
    float invBatch = 1.0f / batchSize;
    for(int i = 0; i < gradc.size(); i++) {
        for(int j = 0; j < gradc[0].size(); j++) {
            gradc[i][j] *= invBatch;
            gradb[i][j] *= invBatch;
        }
    }
}


/**
 * @brief batch layer backprop for mnn hidden layers
 * @param[in] incoming batch of incoming gradients
 * @param[out] outgoing batch of outgoing gradients
 * @param[in] prevAct batch activation of previous layer
 * @param[in] C current layers coefficients weights matrix
 * @param[out] gradc accumulated gradients for C matrix
 * @param[out] gradb accumulated gradients for B matrix
 * @param[in] m order of monomial
 * @param[in] alpha gradient splitting factor
 */
void layerBackwardBatch(const std::vector<std::vector<float>>& incoming,
                    std::vector<std::vector<float>>& outgoing,
                    const std::vector<std::vector<float>>& prevAct,
                    std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb,
                    float m,
                    float alpha)
{
    int batchSize = incoming.size();
    outgoing.resize(batchSize);

    std::vector<std::vector<float>> C_T = transpose(C);

    for(int b = 0; b < batchSize; ++b) {
        std::vector<float> prev_p = power(prevAct[b], m);
        
        // Accumulate gradients
        for(int i = 0; i < prev_p.size(); i++) {
            for(int j = 0; j < incoming[b].size(); j++) {
                gradc[i][j] += alpha * prev_p[i] * incoming[b][j];
                gradb[i][j] += (1.0f - alpha) * incoming[b][j];
            }
        }

        // Compute outgoing gradient for this item
        // dprev_p = m * prevAct^(m-1) = m * prev_p / prevAct
        const float epsilon = 1e-8f;
        std::vector<float> dprev_p(prevAct[b].size());
        for (size_t i = 0; i < prevAct[b].size(); ++i) {
            dprev_p[i] = m * prev_p[i] / (prevAct[b][i] + epsilon);
        }
    
        // dprevAct
        std::vector<float> dprevAct(prevAct[b].size(), 0.0f);
        std::transform(prevAct[b].begin(), prevAct[b].end(), dprevAct.begin(), 
                        [](float x) { return x*(1.0f - x); });

        std::vector<float> out_g = multiply(incoming[b], C_T);
        out_g = multiply(out_g, dprev_p);
        out_g = multiply(out_g, dprevAct);
        outgoing[b] = out_g;
    }

    // Average gradients
    float invBatch = 1.0f / batchSize;
    for(int i = 0; i < gradc.size(); i++) {
        for(int j = 0; j < gradc[0].size(); j++) {
            gradc[i][j] *= invBatch;
            gradb[i][j] *= invBatch;
        }
    }
}

// BATCH BACKPROP FOR MNN2D

/**
 * @brief batch layer backprop for mnn2d for first layer
 * @param[in] incoming batch of incoming gradients (dL/dz_l)
 * @param[in] input input to mnn2d is used here
 * @param[in] C current layers coefficients weights matrix
 * @param[out] gradc accumulated gradients for C matrix
 * @param[out] gradb accumulated gradients for B matrix
 * @param[in] m order of monomial
 * @param[in] alpha gradient splitting factor
 */
void layerBackwardBatch(const std::vector<std::vector<std::vector<float>>>& incoming,
                    const std::vector<std::vector<std::vector<float>>>& input,
                    std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb,
                    float m,
                    float alpha)
{
    // Initialize gradients to 0
    for(auto& row : gradc) std::fill(row.begin(), row.end(), 0.0f);
    for(auto& row : gradb) std::fill(row.begin(), row.end(), 0.0f);

    int batchSize = incoming.size();

    for(int b = 0; b < batchSize; ++b) {
        // dz_l/dB_l = v1^T
        std::vector<std::vector<float>> v1T(input[b][0].size(), std::vector<float>(input[b].size(), 1.0f));
        // dz_l/dC_l
        std::vector<std::vector<float>> prev_p = power(input[b], m);

        std::vector<std::vector<float>> cur_gradc = multiply(transpose(prev_p), incoming[b]);
        std::vector<std::vector<float>> cur_gradb = multiply(v1T, incoming[b]);

        for(int i = 0; i < gradc.size(); i++) {
            for(int j = 0; j < gradc[0].size(); j++) {
                gradc[i][j] += alpha * cur_gradc[i][j];
                gradb[i][j] += (1.0f - alpha) * cur_gradb[i][j];
            }
        }
    }

    // Average gradients
    float invBatch = 1.0f / batchSize;
    for(int i = 0; i < gradc.size(); i++) {
        for(int j = 0; j < gradc[0].size(); j++) {
            gradc[i][j] *= invBatch;
            gradb[i][j] *= invBatch;
        }
    }
}


/**
 * @brief batch layer backprop for mnn2d hidden layers
 * @param[in] incoming batch of incoming gradients
 * @param[out] outgoing batch of outgoing gradients
 * @param[in] dotProds batch of previous layers dot product
 * @param[in] prevAct batch activation of previous layer
 * @param[in] C current layers coefficients weights matrix
 * @param[out] gradc accumulated gradients for C matrix
 * @param[out] gradb accumulated gradients for B matrix
 * @param[in] m order of monomial
 * @param[in] alpha gradient splitting factor
 */
void layerBackwardBatch(const std::vector<std::vector<std::vector<float>>>& incoming,
                    std::vector<std::vector<std::vector<float>>>& outgoing,
                    const std::vector<std::vector<std::vector<float>>>& dotProds,
                    const std::vector<std::vector<std::vector<float>>>& prevAct,
                    std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb,
                    float m,
                    float alpha)
{
    // Initialize gradients to 0
    for(auto& row : gradc) std::fill(row.begin(), row.end(), 0.0f);
    for(auto& row : gradb) std::fill(row.begin(), row.end(), 0.0f);

    int batchSize = incoming.size();
    outgoing.resize(batchSize);

    std::vector<std::vector<float>> C_T = transpose(C);

    for(int b = 0; b < batchSize; ++b) {
        std::vector<std::vector<float>> v1T(prevAct[b][0].size(), std::vector<float>(prevAct[b].size(), 1.0f));
        std::vector<std::vector<float>> prev_p = power(prevAct[b], m);
        
        // Accumulate gradients
        std::vector<std::vector<float>> cur_gradc = multiply(transpose(prev_p), incoming[b]);
        std::vector<std::vector<float>> cur_gradb = multiply(v1T, incoming[b]);

        for(int i = 0; i < gradc.size(); i++) {
            for(int j = 0; j < gradc[0].size(); j++) {
                gradc[i][j] += alpha * cur_gradc[i][j];
                gradb[i][j] += (1.0f - alpha) * cur_gradb[i][j];
            }
        }

        // Compute outgoing gradient
        std::vector<std::vector<float>> dprev_p(prevAct[b].size(), std::vector<float>(prevAct[b][0].size(), 0.0f));
        for (size_t i = 0; i < prev_p.size(); ++i) {
            std::transform(prev_p[i].begin(), prev_p[i].end(), dprev_p[i].begin(),
                        [&m](float x) {
                            float result = m * std::pow(x, m - 1.0f);
                            return std::max(-1e5f, std::min(1e5f, result)); // Clamp to a larger range
                        });
        }
        std::vector<std::vector<float>> dprevAct = reshape(softmaxDer(flatten(dotProds[b])), dotProds[b].size(), dotProds[b][0].size());

        std::vector<std::vector<float>> out_g = multiply(incoming[b], C_T);
        for(int i = 0; i < out_g.size(); i++) {
            for(int j = 0; j < out_g[0].size(); j++) {
                out_g[i][j] = out_g[i][j] * dprev_p[i][j] * dprevAct[i][j];
            }
        }
        outgoing[b] = out_g;
    }

    // Average gradients
    float invBatch = 1.0f / batchSize;
    for(int i = 0; i < gradc.size(); i++) {
        for(int j = 0; j < gradc[0].size(); j++) {
            gradc[i][j] *= invBatch;
            gradb[i][j] *= invBatch;
        }
    }
}

// thread versions
#include <thread>
#include <mutex>

// Helper for thread counts
inline unsigned int get_concurrency(size_t limit) {
    unsigned int num = std::thread::hardware_concurrency();
    return num == 0 ? 2 : std::min(num, static_cast<unsigned int>(limit));
}

/**
 * @brief Threaded batch layer backprop for mnn for first layer
 * Strategy: Parallelize over Input Features (rows of gradc/gradb) to avoid locks.
 */
void layerBackwardBatchThread(const std::vector<std::vector<float>>& incoming,
                              const std::vector<std::vector<float>>& input, 
                              std::vector<std::vector<float>>& C,
                              std::vector<std::vector<float>>& gradc,
                              std::vector<std::vector<float>>& gradb,
                              float m,
                              float alpha)
{
    size_t batchSize = incoming.size();
    if (batchSize == 0) return;
    size_t inSize = input[0].size();
    size_t outSize = incoming[0].size();
    float invBatch = 1.0f / batchSize;

    // 1. Pre-calculate Power (Parallel over Batch)
    std::vector<std::vector<float>> prev_p(batchSize, std::vector<float>(inSize));
    {
        unsigned int nt = get_concurrency(batchSize);
        std::vector<std::thread> pool;
        size_t cs = batchSize / nt;
        auto power_task = [&](size_t s, size_t e) {
            for(size_t b=s; b<e; ++b) prev_p[b] = power(input[b], m);
        };
        for(unsigned t=0; t<nt; ++t) 
            pool.emplace_back(power_task, t*cs, (t==nt-1)?batchSize:(t+1)*cs);
        for(auto& t : pool) t.join();
    }

    // 2. Accumulate Gradients (Parallel over Input Rows 'i')
    // Each thread takes a set of rows and sums the whole batch for those rows.
    // No Mutex needed.
    unsigned int num_threads = get_concurrency(inSize);
    std::vector<std::thread> threads;
    size_t chunk_size = inSize / num_threads;

    auto grad_worker = [&](size_t start_i, size_t end_i) {
        for(size_t i = start_i; i < end_i; ++i) {
            for(size_t j = 0; j < outSize; ++j) {
                float sum_c = 0.0f;
                float sum_b = 0.0f;
                
                // Sum over batch
                for(size_t b = 0; b < batchSize; ++b) {
                    float inc = incoming[b][j];
                    sum_c += prev_p[b][i] * inc;
                    sum_b += inc;
                }

                // Apply alpha and average immediately
                gradc[i][j] = (sum_c * alpha) * invBatch;
                gradb[i][j] = (sum_b * (1.0f - alpha)) * invBatch;
            }
        }
    };

    for(unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? inSize : start + chunk_size;
        threads.emplace_back(grad_worker, start, end);
    }
    for(auto& t : threads) t.join();
}


/**
 * @brief Threaded batch layer backprop for mnn hidden layers
 * Strategy: Two parallel passes. 
 * 1. Parallelize Rows of GradC/GradB (Accumulation).
 * 2. Parallelize Batch Index (Outgoing Calculation).
 */
void layerBackwardBatchThread(const std::vector<std::vector<float>>& incoming,
                              std::vector<std::vector<float>>& outgoing,
                              const std::vector<std::vector<float>>& prevAct,
                              std::vector<std::vector<float>>& C,
                              std::vector<std::vector<float>>& gradc,
                              std::vector<std::vector<float>>& gradb,
                              float m,
                              float alpha)
{
    size_t batchSize = incoming.size();
    if (batchSize == 0) return;
    size_t inSize = prevAct[0].size();     // width[l-1]
    size_t outSize = incoming[0].size();   // width[l]
    float invBatch = 1.0f / batchSize;

    outgoing.resize(batchSize, std::vector<float>(inSize));

    // 1. Pre-calculate Power (Parallel Batch)
    std::vector<std::vector<float>> prev_p(batchSize, std::vector<float>(inSize));
    {
        unsigned int nt = get_concurrency(batchSize);
        std::vector<std::thread> pool;
        size_t cs = batchSize / nt;
        auto power_task = [&](size_t s, size_t e) {
            for(size_t b=s; b<e; ++b) prev_p[b] = power(prevAct[b], m);
        };
        for(unsigned t=0; t<nt; ++t) 
            pool.emplace_back(power_task, t*cs, (t==nt-1)?batchSize:(t+1)*cs);
        for(auto& t : pool) t.join();
    }

    // 2. Gradients (Parallel over Input Rows 'i')
    {
        unsigned int nt = get_concurrency(inSize);
        std::vector<std::thread> threads;
        size_t cs = inSize / nt;

        auto grad_worker = [&](size_t start_i, size_t end_i) {
            for(size_t i = start_i; i < end_i; ++i) {
                for(size_t j = 0; j < outSize; ++j) {
                    float sum_c = 0.0f;
                    float sum_b = 0.0f;
                    for(size_t b = 0; b < batchSize; ++b) {
                        float inc = incoming[b][j];
                        sum_c += prev_p[b][i] * inc;
                        sum_b += inc;
                    }
                    gradc[i][j] = (sum_c * alpha) * invBatch;
                    gradb[i][j] = (sum_b * (1.0f - alpha)) * invBatch;
                }
            }
        };

        for(unsigned t=0; t<nt; ++t) {
            threads.emplace_back(grad_worker, t*cs, (t==nt-1)?inSize:(t+1)*cs);
        }
        for(auto& t : threads) t.join();
    }

    // 3. Outgoing (Parallel over Batch 'b')
    {
        unsigned int nt = get_concurrency(batchSize);
        std::vector<std::thread> threads;
        size_t cs = batchSize / nt;

        auto out_worker = [&](size_t start_b, size_t end_b) {
            const float epsilon = 1e-8f;
            for(size_t b = start_b; b < end_b; ++b) {
                // Calculate vector-matrix product: incoming[b] * C^T 
                // Equivalent to: for each input node i, sum(incoming[j] * C[i][j])
                for(size_t i = 0; i < inSize; ++i) {
                    float dot = 0.0f;
                    for(size_t j = 0; j < outSize; ++j) {
                        dot += incoming[b][j] * C[i][j];
                    }

                    // Derivatives
                    float val = prevAct[b][i];
                    // dprev_p = m * x^(m-1)
                    float d_pow = m * std::pow(val, m - 1.0f); 
                    // dprevAct = x * (1-x)
                    float d_act = val * (1.0f - val);

                    outgoing[b][i] = dot * d_pow * d_act;
                }
            }
        };

        for(unsigned t=0; t<nt; ++t) {
            threads.emplace_back(out_worker, t*cs, (t==nt-1)?batchSize:(t+1)*cs);
        }
        for(auto& t : threads) t.join();
    }
}


/**
 * @brief Threaded batch layer backprop for mnn2d for first layer
 * Strategy: Parallelize over Input Feature columns (gradc rows).
 */
void layerBackwardBatchThread(const std::vector<std::vector<std::vector<float>>>& incoming,
                              const std::vector<std::vector<std::vector<float>>>& input,
                              std::vector<std::vector<float>>& C,
                              std::vector<std::vector<float>>& gradc,
                              std::vector<std::vector<float>>& gradb,
                              float m,
                              float alpha)
{
    // incoming: [Batch][Row][Col] -> Row is spatial, Col is OutFeature
    // input:    [Batch][Row][Col] -> Row is spatial, Col is InFeature
    
    size_t batchSize = incoming.size();
    if(batchSize == 0) return;
    size_t numRows = incoming[0].size();    // Spatial dimension
    size_t inFeatures = input[0][0].size(); // gradc rows
    size_t outFeatures = incoming[0][0].size(); // gradc cols
    float invBatch = 1.0f / batchSize;

    // 1. Pre-calculate Power (Parallel Batch)
    std::vector<std::vector<std::vector<float>>> prev_p(batchSize);
    {
        unsigned int nt = get_concurrency(batchSize);
        std::vector<std::thread> pool;
        size_t cs = batchSize / nt;
        auto power_task = [&](size_t s, size_t e) {
            for(size_t b=s; b<e; ++b) prev_p[b] = power(input[b], m);
        };
        for(unsigned t=0; t<nt; ++t) 
            pool.emplace_back(power_task, t*cs, (t==nt-1)?batchSize:(t+1)*cs);
        for(auto& t : pool) t.join();
    }

    // 2. Gradients (Parallel over InFeatures 'i')
    {
        unsigned int nt = get_concurrency(inFeatures);
        std::vector<std::thread> threads;
        size_t cs = inFeatures / nt;

        auto grad_worker = [&](size_t start_i, size_t end_i) {
            for(size_t i = start_i; i < end_i; ++i) {
                for(size_t j = 0; j < outFeatures; ++j) {
                    float sum_c = 0.0f;
                    float sum_b = 0.0f;
                    
                    // Sum over Batch AND Spatial Rows
                    for(size_t b = 0; b < batchSize; ++b) {
                        for(size_t r = 0; r < numRows; ++r) {
                            float inc = incoming[b][r][j];
                            // prev_p^T * incoming effectively matches [r][i] with [r][j]
                            sum_c += prev_p[b][r][i] * inc;
                            sum_b += inc;
                        }
                    }
                    
                    gradc[i][j] = (sum_c * alpha) * invBatch;
                    gradb[i][j] = (sum_b * (1.0f - alpha)) * invBatch;
                }
            }
        };

        for(unsigned t=0; t<nt; ++t) {
            threads.emplace_back(grad_worker, t*cs, (t==nt-1)?inFeatures:(t+1)*cs);
        }
        for(auto& t : threads) t.join();
    }
}


/**
 * @brief Threaded batch layer backprop for mnn2d hidden layers
 */
void layerBackwardBatchThread(const std::vector<std::vector<std::vector<float>>>& incoming,
                              std::vector<std::vector<std::vector<float>>>& outgoing,
                              const std::vector<std::vector<std::vector<float>>>& dotProds,
                              const std::vector<std::vector<std::vector<float>>>& prevAct,
                              std::vector<std::vector<float>>& C,
                              std::vector<std::vector<float>>& gradc,
                              std::vector<std::vector<float>>& gradb,
                              float m,
                              float alpha)
{
    size_t batchSize = incoming.size();
    if(batchSize == 0) return;
    size_t numRows = prevAct[0].size();
    size_t inFeatures = prevAct[0][0].size();
    size_t outFeatures = incoming[0][0].size();
    float invBatch = 1.0f / batchSize;

    outgoing.resize(batchSize);

    // 1. Pre-calculate Power (Parallel Batch)
    std::vector<std::vector<std::vector<float>>> prev_p(batchSize);
    {
        unsigned int nt = get_concurrency(batchSize);
        std::vector<std::thread> pool;
        size_t cs = batchSize / nt;
        auto power_task = [&](size_t s, size_t e) {
            for(size_t b=s; b<e; ++b) prev_p[b] = power(prevAct[b], m);
        };
        for(unsigned t=0; t<nt; ++t) 
            pool.emplace_back(power_task, t*cs, (t==nt-1)?batchSize:(t+1)*cs);
        for(auto& t : pool) t.join();
    }

    // 2. Gradients (Parallel over InFeatures 'i')
    {
        unsigned int nt = get_concurrency(inFeatures);
        std::vector<std::thread> threads;
        size_t cs = inFeatures / nt;

        auto grad_worker = [&](size_t start_i, size_t end_i) {
            for(size_t i = start_i; i < end_i; ++i) {
                for(size_t j = 0; j < outFeatures; ++j) {
                    float sum_c = 0.0f;
                    float sum_b = 0.0f;
                    // Sum over Batch and Spatial Rows
                    for(size_t b = 0; b < batchSize; ++b) {
                        for(size_t r = 0; r < numRows; ++r) {
                            float inc = incoming[b][r][j];
                            sum_c += prev_p[b][r][i] * inc;
                            sum_b += inc;
                        }
                    }
                    gradc[i][j] = (sum_c * alpha) * invBatch;
                    gradb[i][j] = (sum_b * (1.0f - alpha)) * invBatch;
                }
            }
        };
        for(unsigned t=0; t<nt; ++t) 
            threads.emplace_back(grad_worker, t*cs, (t==nt-1)?inFeatures:(t+1)*cs);
        for(auto& t : threads) t.join();
    }

    // 3. Outgoing (Parallel over Batch 'b')
    {
        unsigned int nt = get_concurrency(batchSize);
        std::vector<std::thread> threads;
        size_t cs = batchSize / nt;

        auto out_worker = [&](size_t start_b, size_t end_b) {
            for(size_t b = start_b; b < end_b; ++b) {
                
                // Helper: dprevAct logic (Sequential within thread, but parallel across batches)
                // Note: reshape/softmaxDer/flatten are expensive. 
                // We assume these functions (mnn2d.hpp) are thread-safe (stateless).
                std::vector<std::vector<float>> dprevAct = reshape(
                    softmaxDer(flatten(dotProds[b])), 
                    dotProds[b].size(), 
                    dotProds[b][0].size()
                );

                outgoing[b].resize(numRows, std::vector<float>(inFeatures));

                for(size_t r = 0; r < numRows; ++r) {
                    for(size_t i = 0; i < inFeatures; ++i) {
                        
                        // Dot Product: incoming[b][r] . C[i] (implicitly C^T row i)
                        float dot = 0.0f;
                        for(size_t j = 0; j < outFeatures; ++j) {
                            dot += incoming[b][r][j] * C[i][j];
                        }

                        // dprev_p
                        float p_val = prevAct[b][r][i];
                        float d_pow = m * std::pow(p_val, m - 1.0f);
                        // Clamp logic from sequential code
                        d_pow = std::max(-1e5f, std::min(1e5f, d_pow));

                        outgoing[b][r][i] = dot * d_pow * dprevAct[r][i];
                    }
                }
            }
        };

        for(unsigned t=0; t<nt; ++t) 
            threads.emplace_back(out_worker, t*cs, (t==nt-1)?batchSize:(t+1)*cs);
        for(auto& t : threads) t.join();
    }
}

#endif