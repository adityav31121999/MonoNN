#include "operators.hpp"
#include <iostream>
#include <limits>

/**
 * @brief Compute statistics (mean, std, min, max) for a 2D vector.
 * @param data Input matrix.
 * @return Statistics struct containing mean, std, min, max.
 */
Statistics computeStats(const std::vector<float>& data) {
    Statistics stats;
    
    // Handle edge cases
    if (data.empty()) {
        stats.mean = 0.0f;
        stats.std = 0.0f;
        stats.min = 0.0f;
        stats.max = 0.0f;
        return stats;
    }

    // Compute statistics
    float sum = 0.0f;
    float sum_sq = 0.0f;
    size_t count = 0;
    stats.min = std::numeric_limits<float>::max();
    stats.max = -std::numeric_limits<float>::max();
    
    for (float val : data) {
        sum += val;
        sum_sq += val * val;
        count++;
        stats.min = std::min(stats.min, val);
        stats.max = std::max(stats.max, val);
    }
    
    // Handle single element case
    if (count == 1) {
        stats.mean = sum;
        stats.std = 0.0f;
        return stats;
    }
    
    stats.mean = sum / count;
    float variance = (sum_sq / count) - (stats.mean * stats.mean);
    stats.std = std::sqrt(std::max(0.0f, variance));
    
    return stats;
}

/**
 * @brief Compute statistics (mean, std, min, max) for a 2D vector.
 * @param data Input matrix.
 * @return Statistics struct containing mean, std, min, max.
 */
Statistics computeStats(const std::vector<std::vector<float>>& data) {
    Statistics stats;
    
    // Handle edge cases
    if (data.empty() || data[0].empty()) {
        stats.mean = 0.0f;
        stats.std = 0.0f;
        stats.min = 0.0f;
        stats.max = 0.0f;
        return stats;
    }
    
    // Compute statistics
    float sum = 0.0f;
    float sum_sq = 0.0f;
    size_t count = 0;
    stats.min = std::numeric_limits<float>::max();
    stats.max = -std::numeric_limits<float>::max();
    
    for (const auto& row : data) {
        for (float val : row) {
            sum += val;
            sum_sq += val * val;
            count++;
            stats.min = std::min(stats.min, val);
            stats.max = std::max(stats.max, val);
        }
    }
    
    // Handle single element case
    if (count == 1) {
        stats.mean = sum;
        stats.std = 0.0f;
        return stats;
    }
    
    stats.mean = sum / count;
    float variance = (sum_sq / count) - (stats.mean * stats.mean);
    stats.std = std::sqrt(std::max(0.0f, variance));
    
    return stats;
}

/**
 * @brief compute statistics for MNN
 * @param cweights coefficients
 * @param bweights biases
 * @param cgrad gradients for coefficients
 * @param bgrad gradients for biases
 * @param act activations
 */
void computeStats(const std::vector<std::vector<std::vector<float>>>& cweights, const std::vector<std::vector<std::vector<float>>>& bweights,
        const std::vector<std::vector<std::vector<float>>>& cgrad, const std::vector<std::vector<std::vector<float>>>& bgrad,
        const std::vector<std::vector<float>>& act)
{
    Statistics stats;    
    // Compute statistics
    float sum = 0.0f;
    float sum_sq = 0.0f;
    size_t count = 0;
    stats.min = std::numeric_limits<float>::max();
    stats.max = -std::numeric_limits<float>::max();
    std::cout << "---------------Analysis---------------\n"; 
    std::cout << "LAYER - PARAM - MAX - MIN - MEAN - STD\n"; 
    for(int i = 0; i < act.size(); i++) {
        std::cout << i << " - ";
        stats = computeStats(cweights[i]);
        std::cout << "C - " << stats.max << " - " << stats.min << " - " << stats.mean << " - " << stats.std << "\n";
        std::cout << i << " - ";
        stats = computeStats(bweights[i]);
        std::cout << "B - " << stats.max << " - " << stats.min << " - " << stats.mean << " - " << stats.std << "\n";
        std::cout << i << " - ";
        stats = computeStats(cgrad[i]);
        std::cout << "Cg - " << stats.max << " - " << stats.min << " - " << stats.mean << " - " << stats.std << "\n";
        std::cout << i << " - ";
        stats = computeStats(bgrad[i]);
        std::cout << "Cg - " << stats.max << " - " << stats.min << " - " << stats.mean << " - " << stats.std << "\n";
        std::cout << i << " - ";
        stats = computeStats(act[i]);
        std::cout << "A - " << stats.max << " - " << stats.min << " - " << stats.mean << " - " << stats.std << "\n";
    }
    std::cout << "--------------------------------------\n";
}

/**
 * @brief compute statistics for MNN2D
 * @param cweights coefficients
 * @param bweights biases
 * @param cgrad gradients for coefficients
 * @param bgrad gradients for biases
 * @param act activations
 */
void computeStats(const std::vector<std::vector<std::vector<float>>>& cweights, const std::vector<std::vector<std::vector<float>>>& bweights,
        const std::vector<std::vector<std::vector<float>>>& cgrad, const std::vector<std::vector<std::vector<float>>>& bgrad,
        const std::vector<std::vector<std::vector<float>>>& act)
{
    Statistics stats;    
    // Compute statistics
    float sum = 0.0f;
    float sum_sq = 0.0f;
    size_t count = 0;
    stats.min = std::numeric_limits<float>::max();
    stats.max = -std::numeric_limits<float>::max();
    std::cout << "---------------Analysis---------------\n"; 
    std::cout << "LAYER - PARAM - MAX - MIN - MEAN - STD\n"; 
    for(int i = 0; i < act.size(); i++) {
        std::cout << i << " - ";
        stats = computeStats(cweights[i]);
        std::cout << "C - " << stats.max << " - " << stats.min << " - " << stats.mean << " - " << stats.std << "\n";
        std::cout << i << " - ";
        stats = computeStats(bweights[i]);
        std::cout << "B - " << stats.max << " - " << stats.min << " - " << stats.mean << " - " << stats.std << "\n";
        std::cout << i << " - ";
        stats = computeStats(cgrad[i]);
        std::cout << "Cg - " << stats.max << " - " << stats.min << " - " << stats.mean << " - " << stats.std << "\n";
        std::cout << i << " - ";
        stats = computeStats(bgrad[i]);
        std::cout << "Cg - " << stats.max << " - " << stats.min << " - " << stats.mean << " - " << stats.std << "\n";
        std::cout << i << " - ";
        stats = computeStats(act[i]);
        std::cout << "A - " << stats.max << " - " << stats.min << " - " << stats.mean << " - " << stats.std << "\n";
    }
    std::cout << "--------------------------------------\n";
}
