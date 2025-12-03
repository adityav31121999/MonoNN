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
 * @param[in] prevAct activation of previous layer
 * @param[in] C current layers coefficients weights matrix
 * @param[out] gradc gradients for C matrix
 * @param[out] gradb gradeitns for B matrix
 * @param[in] m order of monomial
 * @param[in] alpha gradient splitting factor
 */
void layerBackward(const std::vector<float>& incoming,
                    const std::vector<float>& prevAct, 
                    std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb,
                    float m, float alpha)
{
    // std::vector<float> v1(gradb.size(), 1.0f);           // dz_l/dB_l
    std::vector<float> prev_p = power(prevAct, m);          // dz_l/dC_l
    // derivativ of prevAct (sigmoid activated)
    std::vector<float> dprevAct(prevAct.size(), 1.0f);
    std::transform(prevAct.begin(), prevAct.end(), dprevAct.begin(), 
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
                    float m, float alpha)
{
    std::vector<float> prev_p = power(prevAct, m);      // dz_l/dC_l
    // derivative of (prevAct^m) w.r.t prevAct
    std::vector<float> dprev_p(prevAct.size(), 0.0f);   // This is dz_l/da_{l-1} part 1
    std::transform(prevAct.begin(), prevAct.end(), dprev_p.begin(), 
                    [&m](float x) { 
                        float result = m * std::pow(x, m - 1.0f);
                        return std::max(-1e5f, std::min(1e5f, result)); // Clamp to a larger range
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
 * @param[in] prevAct activation of previous layer
 * @param[in] C current layers coefficients weights matrix
 * @param[out] gradc gradients for C matrix
 * @param[out] gradb gradeitns for B matrix
 * @param[in] m order of monomial
 * @param[in] alpha gradient splitting factor
 */
void layerBackward(const std::vector<std::vector<float>>& incoming,
                    const std::vector<std::vector<float>>& prevAct,
                    std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb,
                    float m, float alpha)
{
    // dz_l/dB_l = v1^T
    std::vector<std::vector<float>> v1T(prevAct[0].size(), std::vector<float>(prevAct.size(), 1.0f));
    // dz_l/dC_l
    std::vector<std::vector<float>> prev_p = power(prevAct, m);

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
                    float m, float alpha)
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

#endif