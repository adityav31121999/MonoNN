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
 * @param[in] prevAct batch activation of previous layer
 * @param[in] C current layers coefficients weights matrix
 * @param[out] gradc accumulated gradients for C matrix
 * @param[out] gradb accumulated gradients for B matrix
 * @param[in] m order of monomial
 * @param[in] alpha gradient splitting factor
 */
void layerBackwardBatch(const std::vector<std::vector<float>>& incoming,
                    const std::vector<std::vector<float>>& prevAct, 
                    std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb,
                    float m, float alpha)
{
    // Initialize gradients to 0
    for(auto& row : gradc) std::fill(row.begin(), row.end(), 0.0f);
    for(auto& row : gradb) std::fill(row.begin(), row.end(), 0.0f);

    int batchSize = incoming.size();
    
    for(int b = 0; b < batchSize; ++b) {
        std::vector<float> prev_p = power(prevAct[b], m);
        
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
                    float m, float alpha)
{
    // Initialize gradients to 0
    for(auto& row : gradc) std::fill(row.begin(), row.end(), 0.0f);
    for(auto& row : gradb) std::fill(row.begin(), row.end(), 0.0f);

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
        // dprev_p
        std::vector<float> dprev_p(prevAct[b].size(), 0.0f);
        std::transform(prevAct[b].begin(), prevAct[b].end(), dprev_p.begin(), 
                        [&m](float x) { 
                            float result = m * std::pow(x, m - 1.0f);
                            return std::max(-1e5f, std::min(1e5f, result)); // Clamp to a larger range
                        });
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
 * @param[in] prevAct batch activation of previous layer
 * @param[in] C current layers coefficients weights matrix
 * @param[out] gradc accumulated gradients for C matrix
 * @param[out] gradb accumulated gradients for B matrix
 * @param[in] m order of monomial
 * @param[in] alpha gradient splitting factor
 */
void layerBackwardBatch(const std::vector<std::vector<std::vector<float>>>& incoming,
                    const std::vector<std::vector<std::vector<float>>>& prevAct,
                    std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb,
                    float m, float alpha)
{
    // Initialize gradients to 0
    for(auto& row : gradc) std::fill(row.begin(), row.end(), 0.0f);
    for(auto& row : gradb) std::fill(row.begin(), row.end(), 0.0f);

    int batchSize = incoming.size();

    for(int b = 0; b < batchSize; ++b) {
        // dz_l/dB_l = v1^T
        std::vector<std::vector<float>> v1T(prevAct[b][0].size(), std::vector<float>(prevAct[b].size(), 1.0f));
        // dz_l/dC_l
        std::vector<std::vector<float>> prev_p = power(prevAct[b], m);

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
                    float m, float alpha)
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

#endif