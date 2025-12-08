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
    std::vector<std::vector<float>> dprevAct = reshape(softmaxDer(flatten(dotProds)), dotProds.size(), dotProds[0].size());
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

// Helper to determine thread count
inline unsigned int get_thread_count(size_t work_size) {
    unsigned int num = std::thread::hardware_concurrency();
    return num == 0 ? 2 : std::min(num, static_cast<unsigned int>(work_size));
}

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
 * @brief Threaded single layer backprop for mnn2d for first layer
 * Parallelizes gradient calculation over input features.
 */
void layerBackwardThread(const std::vector<std::vector<float>>& incoming,
                         const std::vector<std::vector<float>>& input,
                         std::vector<std::vector<float>>& C,
                         std::vector<std::vector<float>>& gradc,
                         std::vector<std::vector<float>>& gradb,
                         float m,
                         float alpha)
{
    // input: [Batch x InFeatures]
    // incoming: [Batch x OutFeatures]
    // gradc, gradb: [InFeatures x OutFeatures]

    if (input.empty()) return;

    size_t batch_size = input.size();
    size_t in_features = input[0].size();
    size_t out_features = incoming[0].size();

    // Pre-calculate power (Data parallel on batch)
    // We do this first because random access to prev_p in the next step is heavy
    std::vector<std::vector<float>> prev_p = power(input, m);

    unsigned int num_threads = get_thread_count(in_features);
    std::vector<std::thread> threads;
    size_t chunk_size = in_features / num_threads;

    // Worker computes specific rows of GradC/GradB
    auto worker = [&](size_t start_row, size_t end_row) {
        for(size_t i = start_row; i < end_row; ++i) { // i = in_feature index
            for(size_t j = 0; j < out_features; ++j) { // j = out_feature index
                
                float sum_c = 0.0f;
                float sum_b = 0.0f;

                // Sum over Batch (k)
                // gradc = prev_p^T * incoming -> dot(col i of prev_p, col j of incoming)
                // gradb = v1T * incoming -> sum(col j of incoming)
                for(size_t k = 0; k < batch_size; ++k) {
                    sum_c += prev_p[k][i] * incoming[k][j];
                    sum_b += incoming[k][j]; // v1 is all 1s
                }

                gradc[i][j] = sum_c * alpha;
                gradb[i][j] = sum_b * (1.0f - alpha);
            }
        }
    };

    for(unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? in_features : start + chunk_size;
        threads.emplace_back(worker, start, end);
    }

    for(auto& t : threads) {
        if(t.joinable()) t.join();
    }
}

/**
 * @brief Threaded single layer backprop for mnn2d (Hidden Layer)
 * Splits task into two parallel sections: 1. Gradient Calculation, 2. Outgoing Calculation.
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
    if (prevAct.empty()) return;

    size_t batch_size = prevAct.size();
    size_t in_features = prevAct[0].size();
    size_t out_features = incoming[0].size();

    // --- Preparation ---
    // Calculate derivatives. Using main thread/single threaded for these helpers 
    // to ensure correctness with existing MNN logic (flatten/softmaxDer), 
    // unless those are explicitly thread-safe.
    std::vector<std::vector<float>> prev_p = power(prevAct, m);
    
    // dprev_p calculation (parallelizable element-wise)
    std::vector<std::vector<float>> dprev_p(batch_size, std::vector<float>(in_features));
    {
        unsigned int nt = get_thread_count(batch_size);
        std::vector<std::thread> pool;
        size_t cs = batch_size / nt;
        auto deriv_worker = [&](size_t s, size_t e) {
            for(size_t i=s; i<e; ++i) {
                for(size_t j=0; j<in_features; ++j) {
                    float val = prev_p[i][j]; // This is actually prevAct[i][j]^m, logic check:
                    // Code says: d/dx(x^m) = m * x^(m-1). 
                    // To avoid pow again, if m != 0, x^(m-1) = x^m / x. 
                    // But using pow is safer for stability.
                    float res = m * std::pow(prevAct[i][j], m - 1.0f);
                    dprev_p[i][j] = std::max(-1e5f, std::min(1e5f, res));
                }
            }
        };
        for(unsigned t=0; t<nt; ++t) pool.emplace_back(deriv_worker, t*cs, (t==nt-1)?batch_size:(t+1)*cs);
        for(auto& th : pool) th.join();
    }

    // dprevAct: (softmaxDer(flatten(dotProds))) -> reshape
    // We keep this sequential as it involves global reshape/flatten logic from helper lib
    std::vector<std::vector<float>> dprevAct = reshape(softmaxDer(flatten(dotProds)), batch_size, in_features);


    // --- Parallel Section 1: Gradients (GradC, GradB) ---
    // Parallelize over In-Features (Rows of Grad matrices)
    {
        unsigned int nt = get_thread_count(in_features);
        std::vector<std::thread> grad_threads;
        size_t cs = in_features / nt;

        auto grad_worker = [&](size_t start_row, size_t end_row) {
            for(size_t i = start_row; i < end_row; ++i) {
                for(size_t j = 0; j < out_features; ++j) {
                    float sum_c = 0.0f;
                    float sum_b = 0.0f;
                    for(size_t k = 0; k < batch_size; ++k) {
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
    // Parallelize over Batch Size (Rows of Outgoing)
    // outgoing = (incoming x C^T) . dprev_p . dprevAct
    outgoing.resize(batch_size, std::vector<float>(in_features));
    
    {
        unsigned int nt = get_thread_count(batch_size);
        std::vector<std::thread> out_threads;
        size_t cs = batch_size / nt;

        auto out_worker = [&](size_t start_b, size_t end_b) {
            for(size_t i = start_b; i < end_b; ++i) { // i = batch index
                for(size_t j = 0; j < in_features; ++j) { // j = in feature index
                    
                    // Dot product: row i of incoming * col j of C^T
                    // col j of C^T is row j of C.
                    float dot = 0.0f;
                    for(size_t k = 0; k < out_features; ++k) {
                        dot += incoming[i][k] * C[j][k];
                    }
                    
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