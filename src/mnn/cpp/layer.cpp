#ifdef USE_CPU
#include "mnn.hpp"
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
 */
void layerForward(const std::vector<float>& input, std::vector<float>& output, const std::vector<std::vector<float>>& cweights,
                const std::vector<std::vector<float>>& bweights)
{
    if(input.size() != cweights.size()) {
        throw std::runtime_error("input size and cweights rows mismatch :)");
    }
    if(input.size() != bweights.size()) {
        throw std::runtime_error("input size and bweights rows mismatch :)");
    }
    if(output.size() != cweights[0].size()) {
        throw std::runtime_error("output size and cweights columns mismatch :)");
    }
    if(output.size() != bweights[0].size()) {
        throw std::runtime_error("output size and bweights columns mismatch :)");
    }
    
    for(int i = 0; i < cweights.size(); i++) {
        for(int j = 0; j < cweights[0].size(); j++) {
            // output[j] = sum(input[i]*cweights[i][j] + bweights[i][j]) for j = 0 to height - 1
            output[j] += (input[i]*cweights[i][j]) + bweights[i][j];
        }
    }
}


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
        throw std::runtime_error("input size and cweights rows mismatch :)");
    }
    if(input.size() != bweights.size()) {
        throw std::runtime_error("input size and bweights rows mismatch :)");
    }
    if(output.size() != cweights[0].size()) {
        throw std::runtime_error("output size and cweights columns mismatch :)");
    }
    if(output.size() != bweights[0].size()) {
        throw std::runtime_error("output size and bweights columns mismatch :)");
    }
    
    std::vector<float> powerIn = power(input, n);
    for(int i = 0; i < cweights.size(); i++) {
        for(int j = 0; j < cweights[0].size(); j++) {
            output[j] += (powerIn[i]*cweights[i][j]) + bweights[i][j];
        }
    }
}


/**
 * @brief monomial operation for single layer in forprop
 * @param [in] input matrix input
 * @param [out] output matrix output
 * @param [in] cweights coefficient weights
 * @param [in] bweights bias weights
 */
void layerForward(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output,
                    const std::vector<std::vector<float>>& cweights, const std::vector<std::vector<float>>& bweights)
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
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < cweights[0].size(); ++j) {
            for (size_t k = 0; k < cweights.size(); ++k) {
                output[i][j] += (input[i][k] * cweights[k][j]) + bweights[i][j];
            }
        }
    }
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
            for (size_t k = 0; k < cweights.size(); ++k) {
                output[i][j] += (powerIn[i][k] * cweights[k][j]) + bweights[i][j];
            }
        }
    }
}

//// Backprop ////

/**
 * @brief single layer backprop for mnn for first layer
 * @param[in] incoming incoming gradient (dL/dz_l) vector
 * @param[in] prevAct activation of previous layer vector
 * @param[in, out] C current layers coefficients weights matrix
 * @param[in, out] B current layers bias weights matrix
 * @param[out] gradc gradients for C matrix
 * @param[out] gradb gradeitns for B matrix
 * @param[in] m order of monomial
 * @param[in] learning learning rate
 * @param[in] alpha major gradient for C
 */
void layerBackward(const std::vector<float>& incoming,
                    const std::vector<float>& prevAct, 
                    std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& B,
                    std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb,
                    float m, float alpha, float learning, int typeOfUpdate)
{
    // store final updates here
    std::vector<std::vector<float>> new_C(C.size(), std::vector<float>(C[0].size(), 0.0f));
    std::vector<std::vector<float>> new_B(B.size(), std::vector<float>(B[0].size(), 0.0f));
    new_C = C, new_B = B;

    std::vector<float> v1(B.size(), 1.0f);          // dz_l/dB_l
    std::vector<float> prev_p = power(prevAct, m);  // dz_l/dC_l
    // derivativ of prevAct (no sigmoid applied)
    std::vector<float> dprevAct(prevAct.size(), 1.0f);

    // gradc = alpha * prev_p^T x dl/dz_l, gradc = (1 - alpha) * v1^T x dl/dz_l
    for(int i = 0; i < prev_p.size(); i++) {
        for(int j = 0; j < incoming.size(); j++) {
            gradc[i][j] = alpha * prev_p[i] * incoming[j];
            gradb[i][j] = incoming[j];
        }
    }

    // update weights
    switch (typeOfUpdate) {
        case 0:
            // updateWeights
            updateWeights(new_C, gradc, learning);
            updateWeights(new_B, gradb, learning);
            break;
        case 1:
            // updateWeightsL1
            updateWeightsL1(new_C, gradc, learning, LAMBDA_L1);
            updateWeightsL1(new_B, gradb, learning, LAMBDA_L1);
            break;
        case 2:
            // updateWeightsL2
            updateWeightsL2(new_C, gradc, learning, LAMBDA_L2);
            updateWeightsL2(new_B, gradb, learning, LAMBDA_L2);
            break;
        case 3:
            // updateWeightsElasticNet
            updateWeightsElastic(new_C, gradc, learning, LAMBDA_L1, LAMBDA_L2);
            updateWeightsElastic(new_B, gradb, learning, LAMBDA_L1, LAMBDA_L2);
            break;
        case 4:
            // updateWeightsWeightDecay
            updateWeightsWeightDecay(new_C, gradc, learning, WEIGHT_DECAY);
            updateWeightsWeightDecay(new_B, gradb, learning, WEIGHT_DECAY);
            break;
        case 5:
            // updateWeightsDropout
            updateWeightsDropout(new_C, gradc, learning, DROPOUT_RATE);
            updateWeightsDropout(new_B, gradb, learning, DROPOUT_RATE);
            break;
        default:
            std::cout << "Invalid update type" << std::endl;
            break;
    }

    C = new_C, B = new_B;
}


/**
 * @brief single layer backprop for mnn2d for first layer
 * @param[in] incoming incoming gradient (dL/dz_l) matrix
 * @param[in] prevAct activation of previous layer matrix
 * @param[in, out] C current layers coefficients weights matrix
 * @param[in, out] B current layers bias weights matrix
 * @param[out] gradc gradients for C matrix
 * @param[out] gradb gradeitns for B matrix
 * @param[in] m order of monomial
 * @param[in] learning learning rate
 * @param[in] alpha major gradient for C
 */
void layerBackward(const std::vector<std::vector<float>>& incoming,
                    const std::vector<std::vector<float>>& dotProds,
                    const std::vector<std::vector<float>>& prevAct,
                    std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& B,
                    std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb,
                    float m, float alpha, float learning, int typeOfUpdate)
{
    // store final updates here
    std::vector<std::vector<float>> new_C(C.size(), std::vector<float>(C[0].size(), 0.0f));
    std::vector<std::vector<float>> new_B(B.size(), std::vector<float>(B[0].size(), 0.0f));
    new_C = C, new_B = B;

    std::vector<std::vector<float>> v1(B.size(), std::vector<float>(B[0].size(), 1.0f));          // dz_l/dB_l
    std::vector<std::vector<float>> prev_p = power(prevAct, m);  // dz_l/dC_l
    // derivativ of prevAct (activation is softmax)
    std::vector<std::vector<float>> dprevAct = reshape(softmax(flatten(dotProds)), dotProds.size(), dotProds[0].size());

    // gradc = prev_p^T x dl/dz_l, gradc = (1 - alpha) * v1^T x dl/dz_l
    gradc = multiply(transpose(prev_p), incoming);
    gradb = multiply(transpose(v1), incoming);
    for(int i = 0; i < gradc.size(); i++) {
        for(int j = 0; j < gradc[0].size(); j++) {
            gradc[i][j] = alpha * gradc[i][j];
            gradb[i][j] = (1 - alpha) * gradb[i][j];
        }
    }

    // update weights
    switch (typeOfUpdate) {
        case 0:
            // updateWeights
            updateWeights(new_C, gradc, learning);
            updateWeights(new_B, gradb, learning);
            break;
        case 1:
            // updateWeightsL1
            updateWeightsL1(new_C, gradc, learning, LAMBDA_L1);
            updateWeightsL1(new_B, gradb, learning, LAMBDA_L1);
            break;
        case 2:
            // updateWeightsL2
            updateWeightsL2(new_C, gradc, learning, LAMBDA_L2);
            updateWeightsL2(new_B, gradb, learning, LAMBDA_L2);
            break;
        case 3:
            // updateWeightsElasticNet
            updateWeightsElastic(new_C, gradc, learning, LAMBDA_L1, LAMBDA_L2);
            updateWeightsElastic(new_B, gradb, learning, LAMBDA_L1, LAMBDA_L2);
            break;
        case 4:
            // updateWeightsWeightDecay
            updateWeightsWeightDecay(new_C, gradc, learning, WEIGHT_DECAY);
            updateWeightsWeightDecay(new_B, gradb, learning, WEIGHT_DECAY);
            break;
        case 5:
            // updateWeightsDropout
            updateWeightsDropout(new_C, gradc, learning, DROPOUT_RATE);
            updateWeightsDropout(new_B, gradb, learning, DROPOUT_RATE);
            break;
        default:
            std::cout << "Invalid update type" << std::endl;
            break;
    }

    C = new_C, B = new_B;
}

/**
 * @brief single layer backprop for mnn
 * @param[in] incoming incoming gradient (dL/dz_l) vector
 * @param[out] outgoing outgoing gradient (dL/dz_(l-1)) vector
 * @param[in] prevAct activation of previous layer vector
 * @param[in, out] C current layers coefficients weights matrix
 * @param[in, out] B current layers bias weights matrix
 * @param[out] gradc gradients for C matrix
 * @param[out] gradb gradeitns for B matrix
 * @param[in] m order of monomial
 * @param[in] learning learning rate
 * @param[in] alpha major gradient for C
 */
void layerBackward(const std::vector<float>& incoming, std::vector<float>& outgoing,
                    const std::vector<float>& prevAct, 
                    std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& B,
                    std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb,
                    float m, float alpha, float learning, int typeOfUpdate)
{
    // store final updates here
    std::vector<std::vector<float>> new_C(C.size(), std::vector<float>(C[0].size(), 0.0f));
    std::vector<std::vector<float>> new_B(B.size(), std::vector<float>(B[0].size(), 0.0f));
    new_C = C, new_B = B;

    std::vector<float> v1(B.size(), 1.0f);          // dz_l/dB_l
    std::vector<float> prev_p = power(prevAct, m);  // dz_l/dC_l
    // derivative of prev_p
    std::vector<float> dprev_p(prevAct.size(), 0.0f);
    std::transform(prev_p.begin(), prev_p.end(), dprev_p.begin(), 
                    [&m](float x) { 
                        return m * std::pow(x, m-1.0f); 
                    });
    // derivativ of prevAct
    std::vector<float> dprevAct(prevAct.size(), 0.0f);
    std::transform(prevAct.begin(), prevAct.end(), dprevAct.begin(), 
                    [](float x) { 
                        return x*(1.0f - x); 
                    });

    // outgoing gradient = (dl/dz_l x C^T) . dprev_p . dprevAct
    // incoming gradient x C^T
    std::vector<std::vector<float>> C_T = transpose(C);
    outgoing = multiply(incoming, C_T);
    outgoing = multiply(outgoing, dprev_p);
    outgoing = multiply(outgoing, dprevAct);

    // gradc = alpha * prev_p^T x dl/dz_l, gradc = (1 - alpha) * v1^T x dl/dz_l
    for(int i = 0; i < prev_p.size(); i++) {
        for(int j = 0; j < incoming.size(); j++) {
            gradc[i][j] = alpha * prev_p[i] * incoming[j];
            gradb[i][j] = incoming[j];
        }
    }

    // update weights
    switch (typeOfUpdate) {
        case 0:
            // updateWeights
            updateWeights(new_C, gradc, learning);
            updateWeights(new_B, gradb, learning);
            break;
        case 1:
            // updateWeightsL1
            updateWeightsL1(new_C, gradc, learning, LAMBDA_L1);
            updateWeightsL1(new_B, gradb, learning, LAMBDA_L1);
            break;
        case 2:
            // updateWeightsL2
            updateWeightsL2(new_C, gradc, learning, LAMBDA_L2);
            updateWeightsL2(new_B, gradb, learning, LAMBDA_L2);
            break;
        case 3:
            // updateWeightsElasticNet
            updateWeightsElastic(new_C, gradc, learning, LAMBDA_L1, LAMBDA_L2);
            updateWeightsElastic(new_B, gradb, learning, LAMBDA_L1, LAMBDA_L2);
            break;
        case 4:
            // updateWeightsWeightDecay
            updateWeightsWeightDecay(new_C, gradc, learning, WEIGHT_DECAY);
            updateWeightsWeightDecay(new_B, gradb, learning, WEIGHT_DECAY);
            break;
        case 5:
            // updateWeightsDropout
            updateWeightsDropout(new_C, gradc, learning, DROPOUT_RATE);
            updateWeightsDropout(new_B, gradb, learning, DROPOUT_RATE);
            break;
        default:
            std::cout << "Invalid update type" << std::endl;
            break;
    }

    C = new_C, B = new_B;
}


/**
 * @brief single layer backprop for mnn2d
 * @param[in] incoming incoming gradient (dL/dz_l) matrix
 * @param[out] outgoing outgoing gradient (dL/dz_(l-1)) matrix
 * @param[in] prevAct activation of previous layer matrix
 * @param[in, out] C current layers coefficients weights matrix
 * @param[in, out] B current layers bias weights matrix
 * @param[out] gradc gradients for C matrix
 * @param[out] gradb gradeitns for B matrix
 * @param[in] m order of monomial
 * @param[in] learning learning rate
 * @param[in] alpha major gradient for C
 */
void layerBackward(const std::vector<std::vector<float>>& incoming,
                    std::vector<std::vector<float>>& outgoing,
                    const std::vector<std::vector<float>>& dotProds,
                    const std::vector<std::vector<float>>& prevAct,
                    std::vector<std::vector<float>>& C,
                    std::vector<std::vector<float>>& B,
                    std::vector<std::vector<float>>& gradc,
                    std::vector<std::vector<float>>& gradb,
                    float m, float alpha, float learning, int typeOfUpdate)
{
    // store final updates here
    std::vector<std::vector<float>> new_C(C.size(), std::vector<float>(C[0].size(), 0.0f));
    std::vector<std::vector<float>> new_B(B.size(), std::vector<float>(B[0].size(), 0.0f));
    new_C = C, new_B = B;

    std::vector<std::vector<float>> v1(B.size(), std::vector<float>(B[0].size(), 1.0f));          // dz_l/dB_l
    std::vector<std::vector<float>> prev_p = power(prevAct, m);  // dz_l/dC_l
    // derivative of prev_p (element-wise)
    std::vector<std::vector<float>> dprev_p(prevAct.size(), std::vector<float>(prevAct[0].size(), 0.0f));
    for (size_t i = 0; i < prev_p.size(); ++i) {
        std::transform(prev_p[i].begin(), prev_p[i].end(), dprev_p[i].begin(),
                    [&m](float x) {
                        return m * std::pow(x, m - 1.0f);
                    });
    }
    // derivativ of prevAct (activation is softmax)
    std::vector<std::vector<float>> dprevAct = reshape(softmax(flatten(dotProds)), dotProds.size(), dotProds[0].size());

    // outgoing gradient = (dl/dz_l x C^T) . dprev_p . dprevAct
    // incoming gradient x C^T
    std::vector<std::vector<float>> C_T = transpose(C);
    outgoing = multiply(incoming, C_T);
    for(int i = 0; i < outgoing.size(); i++) {
        for(int j = 0; j < outgoing[0].size(); j++) {
            outgoing[i][j] = outgoing[i][j] * dprev_p[i][j] * dprevAct[i][j];
        }
    }

    // gradc = prev_p^T x dl/dz_l, gradc = (1 - alpha) * v1^T x dl/dz_l
    gradc = multiply(transpose(prev_p), incoming);
    gradb = multiply(transpose(v1), incoming);
    for(int i = 0; i < gradc.size(); i++) {
        for(int j = 0; j < gradc[0].size(); j++) {
            gradc[i][j] = alpha * gradc[i][j];
            gradb[i][j] = (1 - alpha) * gradb[i][j];
        }
    }

    // update weights
    switch (typeOfUpdate) {
        case 0:
            // updateWeights
            updateWeights(new_C, gradc, learning);
            updateWeights(new_B, gradb, learning);
            break;
        case 1:
            // updateWeightsL1
            updateWeightsL1(new_C, gradc, learning, LAMBDA_L1);
            updateWeightsL1(new_B, gradb, learning, LAMBDA_L1);
            break;
        case 2:
            // updateWeightsL2
            updateWeightsL2(new_C, gradc, learning, LAMBDA_L2);
            updateWeightsL2(new_B, gradb, learning, LAMBDA_L2);
            break;
        case 3:
            // updateWeightsElasticNet
            updateWeightsElastic(new_C, gradc, learning, LAMBDA_L1, LAMBDA_L2);
            updateWeightsElastic(new_B, gradb, learning, LAMBDA_L1, LAMBDA_L2);
            break;
        case 4:
            // updateWeightsWeightDecay
            updateWeightsWeightDecay(new_C, gradc, learning, WEIGHT_DECAY);
            updateWeightsWeightDecay(new_B, gradb, learning, WEIGHT_DECAY);
            break;
        case 5:
            // updateWeightsDropout
            updateWeightsDropout(new_C, gradc, learning, DROPOUT_RATE);
            updateWeightsDropout(new_B, gradb, learning, DROPOUT_RATE);
            break;
        default:
            std::cout << "Invalid update type" << std::endl;
            break;
    }

    C = new_C, B = new_B;
}

#endif