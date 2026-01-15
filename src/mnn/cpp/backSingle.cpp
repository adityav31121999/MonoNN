#ifdef USE_CPU
#include "mnn.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>

// BACKPROP FOR MNN

/**
 * @brief single layer backprop for mnn for first layer
 * @param[in] incoming incoming gradient (dL/dz_l) vector
 * @param[in] input input to mnn is used here
 * @param[in] C current layers coefficients weights matrix
 * @param[out] gradc gradients for C matrix
 * @param[out] gradb gradeitns for B matrix
 * @param[in] m order of monomial
 * @param[in] alpha gradient splitting factor
 */
void layerBackward(const std::vector<float>& incoming,
                    const std::vector<float>& input, 
                    std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb,
                    float m,
                    float alpha)
{
    // std::vector<float> v1(gradb.size(), 1.0f);           // dz_l/dB_l
    std::vector<float> prev_p = power(input, m);          // dz_l/dC_l
    // derivativ of input (sigmoid activated)
    std::vector<float> dprevAct(input.size(), 1.0f);
    std::transform(input.begin(), input.end(), dprevAct.begin(), 
                    [](float x) { 
                        return x*(1.0f - x); 
                    }
                );

    // gradc = alpha * prev_p^T x dl/dz_l, gradb = (1 - alpha) * v1^T x dl/dz_l
    for(int i = 0; i < prev_p.size(); i++) {
        for(int j = 0; j < incoming.size(); j++) {
            gradc[i][j] = alpha * prev_p[i] * incoming[j];      // dL/dC_l
            gradb[i][j] = (1.0f - alpha) * incoming[j];         // dL/dB_l
        }
    }
}


/**
 * @brief single layer backprop for mnn
 * @param[in] incoming incoming gradient (dL/dz_l) vector
 * @param[out] outgoing outgoing gradient (dL/dz_(l-1)) vector
 * @param[in] prevAct activation of previous layer
 * @param[in] C current layers coefficients weights matrix
 * @param[out] gradc gradients for C matrix
 * @param[out] gradb gradeitns for B matrix
 * @param[in] m order of monomial
 * @param[in] alpha gradient splitting factor
 */
void layerBackward(const std::vector<float>& incoming,          // width[l]
                    std::vector<float>& outgoing,               // width[l-1]
                    const std::vector<float>& prevAct,          // width[l-1]
                    std::vector<std::vector<float>>& C,         // width[l-1] x width[l]
                    std::vector<std::vector<float>>& gradc,     // width[l-1] x width[l] 
                    std::vector<std::vector<float>>& gradb,     // width[l-1] x width[l]
                    float m,
                    float alpha)
{
    std::vector<float> prev_p = power(prevAct, m);      // dz_l/dC_l
    // derivative of (prevAct^m) w.r.t prevAct
    std::vector<float> dprev_p(prevAct.size(), 0.0f);   // This is dz_l/da_{l-1} part 1
    std::transform(prevAct.begin(), prevAct.end(), dprev_p.begin(), 
                    [&m](float x) { 
                        return (m * std::pow(x, m - 1.0f));
                    });
    // derivativ of prevAct
    std::vector<float> dprevAct(prevAct.size(), 0.0f);
    std::transform(prevAct.begin(), prevAct.end(), dprevAct.begin(), 
                    [](float x) { 
                        return x*(1.0f - x); 
                    });

    // gradc = alpha * prev_p^T x dl/dz_l, gradb = (1 - alpha) * v1^T x dl/dz_l
    // This is an outer product.
    for(int i = 0; i < prev_p.size(); i++) {
        for(int j = 0; j < incoming.size(); j++) {
            gradc[i][j] = alpha * prev_p[i] * incoming[j];  // dL/dC_l
            gradb[i][j] = (1.0f - alpha) * incoming[j];     // dL/dB_l
        }
    }

    // outgoing gradient = (dl/dz_l x C^T) . dprev_p . dprevAct
    // incoming gradient x C^T, C = width[l-1] * width[l]
    std::vector<std::vector<float>> C_T(C[0].size(), std::vector<float>(C.size(), 0.0f));
    C_T = transpose(C);     // width[l] * width[l-1], l = layer count
    outgoing.clear(); outgoing.resize(prevAct.size(), 0.0f);
    outgoing = multiply(incoming, C_T);         // width[l-1]
    outgoing = multiply(outgoing, dprev_p);     // width[l-1], element-wise products
    outgoing = multiply(outgoing, dprevAct);    // width[l-1], element-wise products
}

// BACKPROP FOR MNN2D

/**
 * @brief single layer backprop for mnn2d for first layer
 * @param[in] incoming incoming gradient (dL/dz_l) matrix
 * @param[in] input input to mnn2d is used here
 * @param[in] C current layers coefficients weights matrix
 * @param[out] gradc gradients for C matrix
 * @param[out] gradb gradeitns for B matrix
 * @param[in] m order of monomial
 * @param[in] alpha gradient splitting factor
 */
void layerBackward(const std::vector<std::vector<float>>& incoming,
                    const std::vector<std::vector<float>>& input,
                    std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb,
                    float m,
                    float alpha)
{
    // dz_l/dB_l = v1^T
    std::vector<std::vector<float>> v1T(input[0].size(), std::vector<float>(input.size(), 1.0f));
    // dz_l/dC_l
    std::vector<std::vector<float>> prev_p = power(input, m);

    gradc = multiply(transpose(prev_p), incoming);  // gradc = alpha * prev_p^T x dl/dz_l
    gradb = multiply(v1T, incoming);      // gradb = (1 - alpha) * v1^T x dl/dz_l
    for(int i = 0; i < gradc.size(); i++) {
        for(int j = 0; j < gradc[0].size(); j++) {
            gradc[i][j] *= alpha;          // dL/dC_l
            gradb[i][j] *= (1.0f - alpha); // dL/dB_l
        }
    }
}


/**
 * @brief single layer backprop for mnn2d
 * @param[in] incoming incoming gradient (dL/dz_l) matrix
 * @param[out] outgoing outgoing gradient (dL/dz_(l-1)) matrix
 * @param[in] dotProds previous layers dot product
 * @param[in] prevAct activation of previous layer
 * @param[in] C current layers coefficients weights matrix
 * @param[out] gradc gradients for C matrix
 * @param[out] gradb gradeitns for B matrix
 * @param[in] m order of monomial
 * @param[in] alpha gradient splitting factor
 */
void layerBackward(const std::vector<std::vector<float>>& incoming,
                    std::vector<std::vector<float>>& outgoing,
                    const std::vector<std::vector<float>>& dotProds,
                    const std::vector<std::vector<float>>& prevAct,
                    std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb,
                    float m,
                    float alpha)
{
    // dz_l/dB_l = V1^T
    std::vector<std::vector<float>> v1T(prevAct[0].size(), std::vector<float>(prevAct.size(), 1.0f));
    // dz_l/dC_l
    std::vector<std::vector<float>> prev_p = power(prevAct, m);
    // derivative of prev_p (element-wise)
    std::vector<std::vector<float>> dprev_p(prevAct.size(), std::vector<float>(prevAct[0].size(), 0.0f));
    for (size_t i = 0; i < prev_p.size(); ++i) {
        std::transform(prev_p[i].begin(), prev_p[i].end(), dprev_p[i].begin(),
                    [&m](float x) {
                        float result = m * std::pow(x, m - 1.0f);
                        return std::max(-1e5f, std::min(1e5f, result)); // Clamp to a larger range
                    });
    }
    // derivativ of prevAct (activation is softmax)
    std::vector<std::vector<float>> dprevAct = reluDer(dotProds);
    // gradc = prev_p^T x dl/dz_l, gradc = (1 - alpha) * v1^T x dl/dz_l
    gradc = multiply(transpose(prev_p), incoming);
    gradb = multiply(v1T, incoming);
    for(int i = 0; i < gradc.size(); i++) {
        for(int j = 0; j < gradc[0].size(); j++) {
            gradc[i][j] *= alpha;
            gradb[i][j] *= (1.0f - alpha);
        }
    }

    // outgoing gradient = (dl/dz_l x C^T) . dprev_p . dprevAct
    std::vector<std::vector<float>> C_T = transpose(C);
    outgoing.clear();
    outgoing.resize(dprev_p.size(), std::vector<float>(dprev_p[0].size(), 0.0f));
    outgoing = multiply(incoming, C_T);         // incoming gradient x C^T
    for(int i = 0; i < outgoing.size(); i++) {
        for(int j = 0; j < outgoing[0].size(); j++) {
            outgoing[i][j] = outgoing[i][j] * dprev_p[i][j] * dprevAct[i][j];
        }
    }
}

// thread versions
#include <thread>
#include <mutex>

/**
 * @brief Threaded single layer backprop for mnn for first layer
 * Parallelizes over the rows of the gradient matrices (input size).
 */
void layerBackwardThread(const std::vector<float>& incoming,
                         const std::vector<float>& input, 
                         std::vector<std::vector<float>>& C,
                         std::vector<std::vector<float>>& gradc,
                         std::vector<std::vector<float>>& gradb,
                         float m,
                         float alpha)
{
    // Pre-calculate power input (Read-only for threads)
    std::vector<float> prev_p = power(input, m);
    
    size_t rows = prev_p.size(); // input size
    size_t cols = incoming.size();

    unsigned int num_threads = get_thread_count(rows);
    std::vector<std::thread> threads;
    size_t chunk_size = rows / num_threads;

    auto worker = [&](size_t start_row, size_t end_row) {
        for(size_t i = start_row; i < end_row; ++i) {
            float p_val = prev_p[i];
            for(size_t j = 0; j < cols; ++j) {
                float inc = incoming[j];
                gradc[i][j] = alpha * p_val * inc;
                gradb[i][j] = (1.0f - alpha) * inc;
            }
        }
    };

    for(unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? rows : start + chunk_size;
        threads.emplace_back(worker, start, end);
    }

    for(auto& t : threads) {
        if(t.joinable()) t.join();
    }
}

/**
 * @brief Threaded single layer backprop for mnn (Hidden Layer)
 * Computes Gradients and Outgoing error in parallel splitting over 'prevAct' size (width[l-1]).
 */
void layerBackwardThread(const std::vector<float>& incoming,          
                         std::vector<float>& outgoing,               
                         const std::vector<float>& prevAct,          
                         std::vector<std::vector<float>>& C,         
                         std::vector<std::vector<float>>& gradc,     
                         std::vector<std::vector<float>>& gradb,     
                         float m,
                         float alpha)
{
    size_t prev_size = prevAct.size(); // width[l-1]
    size_t curr_size = incoming.size(); // width[l]

    // Resize outgoing upfront
    outgoing.resize(prev_size);

    unsigned int num_threads = get_thread_count(prev_size);
    std::vector<std::thread> threads;
    size_t chunk_size = prev_size / num_threads;

    auto worker = [&](size_t start_i, size_t end_i) {
        for(size_t i = start_i; i < end_i; ++i) {
            float x = prevAct[i];
            
            // 1. Calculate derivatives locally
            // dprev_p = m * x^(m-1)
            float d_pow = m * std::pow(x, m - 1.0f);
            // dprevAct = x * (1 - x) (assuming sigmoid/logistic behavior from sequential code)
            float d_act = x * (1.0f - x);
            
            // prev_p = x^m
            float p_val = std::pow(x, m);

            float outgoing_sum = 0.0f;

            // 2. Compute Gradients and Outgoing Accumulation
            for(size_t j = 0; j < curr_size; ++j) {
                float inc = incoming[j];
                
                // Update Gradients
                gradc[i][j] = alpha * p_val * inc;
                gradb[i][j] = (1.0f - alpha) * inc;

                // Accumulate Outgoing: incoming[j] * C[i][j]
                // (This implicitly handles the C^T multiplication without allocating C^T)
                outgoing_sum += inc * C[i][j];
            }

            // 3. Finalize Outgoing
            outgoing[i] = outgoing_sum * d_pow * d_act;
        }
    };

    for(unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? prev_size : start + chunk_size;
        threads.emplace_back(worker, start, end);
    }

    for(auto& t : threads) {
        if(t.joinable()) t.join();
    }
}


/**
 * @brief MULTITHREADED layer backprop for mnn2d for first layer
 * Parallelizes gradient computation across input features using thread pool
 * @param[in] incoming incoming gradient (dL/dz_l) matrix [batch × out_features]
 * @param[in] input input to mnn2d [batch × in_features]
 * @param[in] C current layers coefficients weights matrix [in_features × out_features]
 * @param[out] gradc gradients for C matrix [in_features × out_features]
 * @param[out] gradb gradients for B matrix [in_features × out_features]
 * @param[in] m order of monomial
 * @param[in] alpha gradient splitting factor
 * 
 * Thread Strategy: Splits in_features rows across worker threads for parallel gradient accumulation
 */
void layerBackwardThread(const std::vector<std::vector<float>>& incoming,
                         const std::vector<std::vector<float>>& input,
                         std::vector<std::vector<float>>& C,
                         std::vector<std::vector<float>>& gradc,
                         std::vector<std::vector<float>>& gradb,
                         float m,
                         float alpha)
{
    // CRITICAL: Validate dimensions FIRST
    if (input.empty() || incoming.empty()) {
        std::cerr << "ERROR: Empty input or incoming matrix" << std::endl;
        return;
    }

    size_t batch_size = input.size();
    size_t in_features = input[0].size();
    size_t out_features = incoming[0].size();

    // Validate all input rows have same size
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i].size() != in_features) {
            std::cerr << "ERROR: Inconsistent input row sizes at row " << i << std::endl;
            return;
        }
    }
    
    // Validate all incoming rows have same size
    for (size_t i = 0; i < incoming.size(); ++i) {
        if (incoming[i].size() != out_features) {
            std::cerr << "ERROR: Inconsistent incoming row sizes at row " << i << std::endl;
            return;
        }
    }

    // Validate gradient matrix dimensions
    if (gradc.size() != in_features || gradc[0].size() != out_features) {
        std::cerr << "ERROR: gradc dimension mismatch. Expected [" 
                  << in_features << "x" << out_features << "], got ["
                  << gradc.size() << "x" << gradc[0].size() << "]" << std::endl;
        return;
    }
    
    if (gradb.size() != in_features || gradb[0].size() != out_features) {
        std::cerr << "ERROR: gradb dimension mismatch" << std::endl;
        return;
    }

    // Pre-calculate power with error checking
    std::vector<std::vector<float>> prev_p;
    try {
        prev_p = power(input, m);
        
        // Validate prev_p dimensions
        if (prev_p.size() != batch_size || prev_p[0].size() != in_features) {
            std::cerr << "ERROR: power() returned wrong dimensions" << std::endl;
            return;
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR in power() calculation: " << e.what() << std::endl;
        return;
    }

    // Determine thread count
    unsigned int num_threads = get_thread_count(in_features);
    if (num_threads == 0) num_threads = 1;
    
    std::vector<std::thread> threads;
    size_t chunk_size = in_features / num_threads;

    // Worker computes specific rows of GradC/GradB
    auto worker = [&](size_t start_row, size_t end_row) {
        try {
            for(size_t i = start_row; i < end_row; ++i) { // i = in_feature index
                // Bounds check
                if (i >= in_features) continue;
                
                for(size_t j = 0; j < out_features; ++j) { // j = out_feature index
                    float sum_c = 0.0f;
                    float sum_b = 0.0f;

                    // Sum over Batch (k)
                    for(size_t k = 0; k < batch_size; ++k) {
                        // Bounds check before access
                        if (k >= prev_p.size() || i >= prev_p[k].size()) {
                            std::cerr << "ERROR: prev_p out of bounds [" << k << "][" << i << "]" << std::endl;
                            continue;
                        }
                        if (k >= incoming.size() || j >= incoming[k].size()) {
                            std::cerr << "ERROR: incoming out of bounds [" << k << "][" << j << "]" << std::endl;
                            continue;
                        }
                        
                        sum_c += prev_p[k][i] * incoming[k][j];
                        sum_b += incoming[k][j];
                    }

                    gradc[i][j] = sum_c * alpha;
                    gradb[i][j] = sum_b * (1.0f - alpha);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Thread exception in worker: " << e.what() << std::endl;
        }
    };

    // Launch threads
    for(unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? in_features : start + chunk_size;
        threads.emplace_back(worker, start, end);
    }

    // Join all threads
    for(auto& t : threads) {
        if(t.joinable()) t.join();
    }
}

/**
 * @brief MULTITHREADED layer backprop for mnn2d (Hidden Layer)
 * Parallelizes both gradient computation and outgoing error propagation
 * @param[in] incoming incoming gradient (dL/dz_l) matrix [batch × out_features]
 * @param[out] outgoing outgoing gradient (dL/dz_(l-1)) matrix [batch × in_features]
 * @param[in] dotProds previous layers dot product [batch × in_features]
 * @param[in] prevAct activation of previous layer [batch × in_features]
 * @param[in] C current layers coefficients weights matrix [in_features × out_features]
 * @param[out] gradc gradients for C matrix [in_features × out_features]
 * @param[out] gradb gradients for B matrix [in_features × out_features]
 * @param[in] m order of monomial
 * @param[in] alpha gradient splitting factor
 * 
 * Thread Strategy: 
 * - Phase 1: Parallel derivative computation across batch samples
 * - Phase 2: Parallel gradient accumulation across in_features
 * - Phase 3: Parallel outgoing gradient computation across batch samples
 */
void layerBackwardThread(const std::vector<std::vector<float>>& incoming,
                         std::vector<std::vector<float>>& outgoing,
                         const std::vector<std::vector<float>>& dotProds,
                         const std::vector<std::vector<float>>& prevAct,
                         std::vector<std::vector<float>>& C,
                         std::vector<std::vector<float>>& gradc,
                         std::vector<std::vector<float>>& gradb,
                         float m,
                         float alpha)
{
    // CRITICAL: Validate dimensions FIRST
    if (prevAct.empty() || incoming.empty()) {
        std::cerr << "ERROR: Empty prevAct or incoming matrix" << std::endl;
        return;
    }

    size_t batch_size = prevAct.size();
    size_t in_features = prevAct[0].size();
    size_t out_features = incoming[0].size();

    // Validate dimensions
    if (C.size() != in_features || C[0].size() != out_features) {
        std::cerr << "ERROR: C dimension mismatch. Expected [" 
                  << in_features << "x" << out_features << "], got ["
                  << C.size() << "x" << (C.empty() ? 0 : C[0].size()) << "]" << std::endl;
        return;
    }

    // --- Preparation ---
    std::vector<std::vector<float>> prev_p;
    try {
        prev_p = power(prevAct, m);
        if (prev_p.size() != batch_size || prev_p[0].size() != in_features) {
            std::cerr << "ERROR: prev_p dimension mismatch after power()" << std::endl;
            return;
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR in power(prevAct, m): " << e.what() << std::endl;
        return;
    }
    
    // dprev_p calculation (parallelizable element-wise)
    std::vector<std::vector<float>> dprev_p(batch_size, std::vector<float>(in_features));
    {
        unsigned int nt = get_thread_count(batch_size);
        if (nt == 0) nt = 1;
        std::vector<std::thread> pool;
        size_t cs = batch_size / nt;
        
        auto deriv_worker = [&](size_t s, size_t e) {
            for(size_t i=s; i<e && i<batch_size; ++i) {
                for(size_t j=0; j<in_features; ++j) {
                    // Safe bounds check
                    if (i >= prevAct.size() || j >= prevAct[i].size()) continue;
                    
                    float val = prevAct[i][j];
                    float res = m * std::pow(val, m - 1.0f);
                    
                    // Clamp to prevent overflow
                    dprev_p[i][j] = std::max(-1e5f, std::min(1e5f, res));
                }
            }
        };
        
        for(unsigned t=0; t<nt; ++t) {
            size_t start = t*cs;
            size_t end = (t==nt-1) ? batch_size : (t+1)*cs;
            pool.emplace_back(deriv_worker, start, end);
        }
        for(auto& th : pool) if(th.joinable()) th.join();
    }

    // dprevAct: (softmaxDer(flatten(dotProds))) -> reshape
    std::vector<std::vector<float>> dprevAct;
    try {
        dprevAct = reluDer(dotProds);
    } catch (const std::exception& e) {
        std::cerr << "ERROR in softmaxDer/reshape: " << e.what() << std::endl;
        return;
    }

    // --- Parallel Section 1: Gradients (GradC, GradB) ---
    {
        unsigned int nt = get_thread_count(in_features);
        if (nt == 0) nt = 1;
        std::vector<std::thread> grad_threads;
        size_t cs = in_features / nt;

        auto grad_worker = [&](size_t start_row, size_t end_row) {
            for(size_t i = start_row; i < end_row && i < in_features; ++i) {
                for(size_t j = 0; j < out_features; ++j) {
                    float sum_c = 0.0f;
                    float sum_b = 0.0f;
                    
                    for(size_t k = 0; k < batch_size; ++k) {
                        // Bounds checks
                        if (k >= prev_p.size() || i >= prev_p[k].size()) continue;
                        if (k >= incoming.size() || j >= incoming[k].size()) continue;
                        
                        sum_c += prev_p[k][i] * incoming[k][j];
                        sum_b += incoming[k][j];
                    }
                    
                    gradc[i][j] = sum_c * alpha;
                    gradb[i][j] = sum_b * (1.0f - alpha);
                }
            }
        };

        for(unsigned t=0; t<nt; ++t) {
            size_t s = t*cs;
            size_t e = (t==nt-1) ? in_features : s+cs;
            grad_threads.emplace_back(grad_worker, s, e);
        }
        for(auto& th : grad_threads) if(th.joinable()) th.join();
    }

    // --- Parallel Section 2: Outgoing Gradients ---
    outgoing.clear();
    outgoing.resize(batch_size, std::vector<float>(in_features, 0.0f));
    
    {
        unsigned int nt = get_thread_count(batch_size);
        if (nt == 0) nt = 1;
        std::vector<std::thread> out_threads;
        size_t cs = batch_size / nt;

        auto out_worker = [&](size_t start_b, size_t end_b) {
            for(size_t i = start_b; i < end_b && i < batch_size; ++i) {
                for(size_t j = 0; j < in_features; ++j) {
                    float dot = 0.0f;
                    
                    for(size_t k = 0; k < out_features; ++k) {
                        // Bounds checks
                        if (i >= incoming.size() || k >= incoming[i].size()) continue;
                        if (j >= C.size() || k >= C[j].size()) continue;
                        
                        dot += incoming[i][k] * C[j][k];
                    }
                    
                    // Additional bounds checks for dprev_p and dprevAct
                    if (i >= dprev_p.size() || j >= dprev_p[i].size()) continue;
                    if (i >= dprevAct.size() || j >= dprevAct[i].size()) continue;
                    
                    outgoing[i][j] = dot * dprev_p[i][j] * dprevAct[i][j];
                }
            }
        };

        for(unsigned t=0; t<nt; ++t) {
            size_t s = t*cs;
            size_t e = (t==nt-1) ? batch_size : s+cs;
            out_threads.emplace_back(out_worker, s, e);
        }
        for(auto& th : out_threads) if(th.joinable()) th.join();
    }
}

#endif