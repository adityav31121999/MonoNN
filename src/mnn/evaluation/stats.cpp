#include "progress.hpp"
#include <iostream>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <algorithm>


/**
 * @brief get regression scores
 */
void getScore(const std::vector<float>& actual, const std::vector<float>& pred, double SST, double SSR, double SSE) {
    if (actual.size() != pred.size()) {
        throw std::runtime_error("getScore: Sizes of vector must match: " + std::to_string(actual.size()) + " vs. " + std::to_string(pred.size()));
    }

    float m1 = 0.0f;
    float m2 = 0.0f;
    m1 = std::accumulate(actual.begin(), actual.end(), 0.0f);
    m2 = std::accumulate(pred.begin(), pred.end(), 0.0f);

    for(int i = 0; i < actual.size(); i++) {
        SSE += static_cast<double>((actual[i] - pred[i]) * (actual[i] - pred[i]));
        SSR += static_cast<double>((actual[i] - m2) * (actual[i] - m2));
        SST += static_cast<double>((actual[i] - m1) * (actual[i] - m1));
    }
}


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
    stats.std = std::sqrt(std::max<float>(0.0f, variance));
    
    return stats;
}

/**
 * @brief Compute statistics (mean, std, min, max) for a 2D vector.
 * @param data Input matrix.
 * @return Statistics struct containing mean, std, min, max.
 */
Statistics computeStats(const std::vector<std::vector<std::vector<float>>>& batch) {
    Statistics stats;
    
    // Handle edge cases
    if (batch.empty() || batch[0].empty()) {
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
    for(const auto& data: batch) {
        for (const auto& row : data) {
            for (float val : row) {
                sum += val;
                sum_sq += val * val;
                count++;
                stats.min = std::min<float>(stats.min, val);
                stats.max = std::max<float>(stats.max, val);
            }
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
    stats.std = std::sqrt(std::max<float>(0.0f, variance));
    
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
            stats.min = std::min<float>(stats.min, val);
            stats.max = std::max<float>(stats.max, val);
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
    stats.std = std::sqrt(std::max<float>(0.0f, variance));
    
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
void computeStats(const std::vector<std::vector<std::vector<float>>>& cweights,
                  const std::vector<std::vector<std::vector<float>>>& bweights,
                  const std::vector<std::vector<std::vector<float>>>& cgrad,
                  const std::vector<std::vector<std::vector<float>>>& bgrad,
                  const std::vector<std::vector<float>>& act)
{
    std::vector<Statistics> cwstats(cweights.size());
    std::vector<Statistics> bwstats(bweights.size());
    std::vector<Statistics> cgstats(cgrad.size());
    std::vector<Statistics> bgstats(bgrad.size());
    std::vector<Statistics> actstats(act.size());
    // Compute statistics
    float sum = 0.0f;
    float sum_sq = 0.0f;
    size_t count = 0;
    // set min and max
    for(int i = 0; i < act.size(); i++) {
        cwstats[i].min = std::numeric_limits<float>::max();
        cwstats[i].max = -std::numeric_limits<float>::max();
        bwstats[i].min = std::numeric_limits<float>::max();
        bwstats[i].max = -std::numeric_limits<float>::max();
        cgstats[i].min = std::numeric_limits<float>::max();
        cgstats[i].max = -std::numeric_limits<float>::max();
        bgstats[i].min = std::numeric_limits<float>::max();
        bgstats[i].max = -std::numeric_limits<float>::max();
        actstats[i].min = std::numeric_limits<float>::max();
        actstats[i].max = -std::numeric_limits<float>::max();
    }
    // compute stats
    for(int i = 0; i < act.size(); i++) {
        cwstats[i] = computeStats(cweights[i]);
        bwstats[i] = computeStats(bweights[i]);
        cgstats[i] = computeStats(cgrad[i]);
        bgstats[i] = computeStats(bgrad[i]);
        actstats[i] = computeStats(act[i]);
    }

    // print stats
    std::cout << "\n--------------- Analysis (layer: maximum, minimum, mean, std. deviation) ---------------";
    std::cout << "\nCWEIGHTS  :";   for(int i = 0; i < act.size(); i++) std::cout << "\n\t(" << i << " : " << cwstats[i].max << ", " << cwstats[i].min << ", " << cwstats[i].mean << ", " << cwstats[i].std << ")";
    std::cout << "\nBWEIGHTS  :";   for(int i = 0; i < act.size(); i++) std::cout << "\n\t(" << i << " : " << bwstats[i].max << ", " << bwstats[i].min << ", " << bwstats[i].mean << ", " << bwstats[i].std << ")";
    std::cout << "\nCGRAD     :";   for(int i = 0; i < act.size(); i++) std::cout << "\n\t(" << i << " : " << cgstats[i].max << ", " << cgstats[i].min << ", " << cgstats[i].mean << ", " << cgstats[i].std << ")";
    std::cout << "\nBGRAD     :";   for(int i = 0; i < act.size(); i++) std::cout << "\n\t(" << i << " : " << bgstats[i].max << ", " << bgstats[i].min << ", " << bgstats[i].mean << ", " << bgstats[i].std << ")";
    std::cout << "\nACT       :";   for(int i = 0; i < act.size(); i++) std::cout << "\n\t(" << i << " : " << actstats[i].max << ", " << actstats[i].min << ", " << actstats[i].mean << ", " << actstats[i].std << ")";
    std::cout << "\n----------------------------------------------------------------------------------------\n";
}

/**
 * @brief compute statistics for MNN bactch and MNN2D
 * @param cweights coefficients
 * @param bweights biases
 * @param cgrad gradients for coefficients
 * @param bgrad gradients for biases
 * @param act activations
 */
void computeStats(const std::vector<std::vector<std::vector<float>>>& cweights,
                  const std::vector<std::vector<std::vector<float>>>& bweights,
                  const std::vector<std::vector<std::vector<float>>>& cgrad,
                  const std::vector<std::vector<std::vector<float>>>& bgrad,
                  const std::vector<std::vector<std::vector<float>>>& act)
{
    std::vector<Statistics> cwstats(cweights.size());
    std::vector<Statistics> bwstats(bweights.size());
    std::vector<Statistics> cgstats(cgrad.size());
    std::vector<Statistics> bgstats(bgrad.size());
    std::vector<Statistics> actstats(act.size());
    // Compute statistics
    float sum = 0.0f;
    float sum_sq = 0.0f;
    size_t count = 0;
    // set min and max
    for(int i = 0; i < act.size(); i++) {
        cwstats[i].min = std::numeric_limits<float>::max();
        cwstats[i].max = -std::numeric_limits<float>::max();
        bwstats[i].min = std::numeric_limits<float>::max();
        bwstats[i].max = -std::numeric_limits<float>::max();
        cgstats[i].min = std::numeric_limits<float>::max();
        cgstats[i].max = -std::numeric_limits<float>::max();
        bgstats[i].min = std::numeric_limits<float>::max();
        bgstats[i].max = -std::numeric_limits<float>::max();
        actstats[i].min = std::numeric_limits<float>::max();
        actstats[i].max = -std::numeric_limits<float>::max();
    }
    // compute stats
    for(int i = 0; i < act.size(); i++) {
        cwstats[i] = computeStats(cweights[i]);
        bwstats[i] = computeStats(bweights[i]);
        cgstats[i] = computeStats(cgrad[i]);
        bgstats[i] = computeStats(bgrad[i]);
        actstats[i] = computeStats(act[i]);
    }

    // print stats
    std::cout << "\n--------------- Analysis (layer: maximum, minimum, mean, std. deviation) ---------------";
    std::cout << "\nCWEIGHTS  :";   for(int i = 0; i < act.size(); i++) std::cout << "\n\t(" << i << " : " << cwstats[i].max << ", " << cwstats[i].min << ", " << cwstats[i].mean << ", " << cwstats[i].std << ")";
    std::cout << "\nBWEIGHTS  :";   for(int i = 0; i < act.size(); i++) std::cout << "\n\t(" << i << " : " << bwstats[i].max << ", " << bwstats[i].min << ", " << bwstats[i].mean << ", " << bwstats[i].std << ")";
    std::cout << "\nCGRAD     :";   for(int i = 0; i < act.size(); i++) std::cout << "\n\t(" << i << " : " << cgstats[i].max << ", " << cgstats[i].min << ", " << cgstats[i].mean << ", " << cgstats[i].std << ")";
    std::cout << "\nBGRAD     :";   for(int i = 0; i < act.size(); i++) std::cout << "\n\t(" << i << " : " << bgstats[i].max << ", " << bgstats[i].min << ", " << bgstats[i].mean << ", " << bgstats[i].std << ")";
    std::cout << "\nACT       :";   for(int i = 0; i < act.size(); i++) std::cout << "\n\t(" << i << " : " << actstats[i].max << ", " << actstats[i].min << ", " << actstats[i].mean << ", " << actstats[i].std << ")";
    std::cout << "\n----------------------------------------------------------------------------------------\n";
}


/**
 * @brief compute statistics for MNN2D batch
 * @param cweights coefficients
 * @param bweights biases
 * @param cgrad gradients for coefficients
 * @param bgrad gradients for biases
 * @param act activations
 */
void computeStats(const std::vector<std::vector<std::vector<float>>>& cweights,
                  const std::vector<std::vector<std::vector<float>>>& bweights,
                  const std::vector<std::vector<std::vector<float>>>& cgrad,
                  const std::vector<std::vector<std::vector<float>>>& bgrad,
                  const std::vector<std::vector<std::vector<std::vector<float>>>>& act)
{
    std::vector<Statistics> cwstats(cweights.size());
    std::vector<Statistics> bwstats(bweights.size());
    std::vector<Statistics> cgstats(cgrad.size());
    std::vector<Statistics> bgstats(bgrad.size());
    std::vector<Statistics> actstats(act.size());
    // Compute statistics
    float sum = 0.0f;
    float sum_sq = 0.0f;
    size_t count = 0;
    // set min and max
    for(int i = 0; i < act.size(); i++) {
        cwstats[i].min = std::numeric_limits<float>::max();
        cwstats[i].max = -std::numeric_limits<float>::max();
        bwstats[i].min = std::numeric_limits<float>::max();
        bwstats[i].max = -std::numeric_limits<float>::max();
        cgstats[i].min = std::numeric_limits<float>::max();
        cgstats[i].max = -std::numeric_limits<float>::max();
        bgstats[i].min = std::numeric_limits<float>::max();
        bgstats[i].max = -std::numeric_limits<float>::max();
        actstats[i].min = std::numeric_limits<float>::max();
        actstats[i].max = -std::numeric_limits<float>::max();
    }
    // compute stats
    for(int i = 0; i < act.size(); i++) {
        cwstats[i] = computeStats(cweights[i]);
        bwstats[i] = computeStats(bweights[i]);
        cgstats[i] = computeStats(cgrad[i]);
        bgstats[i] = computeStats(bgrad[i]);
        actstats[i] = computeStats(act[i]);
    }

    // print stats
    std::cout << "\n--------------- Analysis (layer: maximum, minimum, mean, std. deviation) ---------------";
    std::cout << "\nCWEIGHTS  :";   for(int i = 0; i < act.size(); i++) std::cout << "\n\t(" << i << " : " << cwstats[i].max << ", " << cwstats[i].min << ", " << cwstats[i].mean << ", " << cwstats[i].std << ")";
    std::cout << "\nBWEIGHTS  :";   for(int i = 0; i < act.size(); i++) std::cout << "\n\t(" << i << " : " << bwstats[i].max << ", " << bwstats[i].min << ", " << bwstats[i].mean << ", " << bwstats[i].std << ")";
    std::cout << "\nCGRAD     :";   for(int i = 0; i < act.size(); i++) std::cout << "\n\t(" << i << " : " << cgstats[i].max << ", " << cgstats[i].min << ", " << cgstats[i].mean << ", " << cgstats[i].std << ")";
    std::cout << "\nBGRAD     :";   for(int i = 0; i < act.size(); i++) std::cout << "\n\t(" << i << " : " << bgstats[i].max << ", " << bgstats[i].min << ", " << bgstats[i].mean << ", " << bgstats[i].std << ")";
    std::cout << "\nACT       :";   for(int i = 0; i < act.size(); i++) std::cout << "\n\t(" << i << " : " << actstats[i].max << ", " << actstats[i].min << ", " << actstats[i].mean << ", " << actstats[i].std << ")";
    std::cout << "\n----------------------------------------------------------------------------------------\n";
}


/**
 * @brief compute statistics for MNN2D batch
 * @param cweights coefficients
 * @param bweights biases
 * @param cgrad gradients for coefficients
 * @param bgrad gradients for biases
 * @param stats statistics (maximum, minimum, mean, standard deviation)
 */
void computeStatsForCsv(const std::vector<std::vector<std::vector<float>>> &cweights, 
                        const std::vector<std::vector<std::vector<float>>> &bweights,
                        const std::vector<std::vector<std::vector<float>>> &cgrad,
                        const std::vector<std::vector<std::vector<float>>> &bgrad,
                        std::vector<std::vector<float>> &stats)
{
    std::vector<Statistics> cwstats(cweights.size());
    std::vector<Statistics> bwstats(bweights.size());
    std::vector<Statistics> cgstats(cgrad.size());
    std::vector<Statistics> bgstats(bgrad.size());
    stats.clear();
    stats.resize(cweights.size() * 4, std::vector<float>(4, 0.0f));
    // Compute statistics
    float sum = 0.0f;
    float sum_sq = 0.0f;
    size_t count = 0;
    // set min and max
    for(int i = 0; i < cweights.size(); i++) {
        cwstats[i].min = std::numeric_limits<float>::max();
        cwstats[i].max = -std::numeric_limits<float>::max();
        bwstats[i].min = std::numeric_limits<float>::max();
        bwstats[i].max = -std::numeric_limits<float>::max();
        cgstats[i].min = std::numeric_limits<float>::max();
        cgstats[i].max = -std::numeric_limits<float>::max();
        bgstats[i].min = std::numeric_limits<float>::max();
        bgstats[i].max = -std::numeric_limits<float>::max();
    }
    // compute stats
    for(int i = 0; i < cweights.size(); i++) {
        cwstats[i] = computeStats(cweights[i]);
        bwstats[i] = computeStats(bweights[i]);
        cgstats[i] = computeStats(cgrad[i]);
        bgstats[i] = computeStats(bgrad[i]);
    }
    for(int i = 0; i < cweights.size(); i++) {
        stats[i] = {cwstats[i].max, cwstats[i].min, cwstats[i].mean, cwstats[i].std};
    }
    for(int i = 0; i < bweights.size(); i++) {
        stats[cweights.size() + i] = {bwstats[i].max, bwstats[i].min, bwstats[i].mean, bwstats[i].std};
    }
    for(int i = 0; i < cgstats.size(); i++) {
        stats[(cweights.size() * 2) + i] = {cgstats[i].max, cgstats[i].min, cgstats[i].mean, cgstats[i].std};
    }
    for(int i = 0; i < bgstats.size(); i++) {
        stats[(cweights.size() * 3) + i] = {bgstats[i].max, bgstats[i].min, bgstats[i].mean, bgstats[i].std};
    }
}


/**
 * @brief compute statistics for MNN2D batch
 * @param cweights coefficients
 * @param bweights biases
 * @param cgrad gradients for coefficients
 * @param bgrad gradients for biases
 * @param stats statistics (maximum, minimum, mean, standard deviation)
 */
void computeStatsForCsv(const std::vector<std::vector<std::vector<float>>> &cweights, 
                        const std::vector<std::vector<std::vector<float>>> &bweights,
                        const std::vector<std::vector<std::vector<float>>> &cgrad,
                        const std::vector<std::vector<std::vector<float>>> &bgrad,
                        const std::vector<std::vector<float>> &activations,
                        std::vector<std::vector<float>> &stats)
{
    std::vector<Statistics> cwstats(cweights.size());
    std::vector<Statistics> bwstats(bweights.size());
    std::vector<Statistics> cgstats(cgrad.size());
    std::vector<Statistics> bgstats(bgrad.size());
    std::vector<Statistics> actstats(activations.size());
    stats.clear();
    stats.resize(cweights.size() * 5, std::vector<float>(4, 0.0f));
    // Compute statistics
    float sum = 0.0f;
    float sum_sq = 0.0f;
    size_t count = 0;
    // set min and max
    for(int i = 0; i < cweights.size(); i++) {
        cwstats[i].min = std::numeric_limits<float>::max();
        cwstats[i].max = -std::numeric_limits<float>::max();
        bwstats[i].min = std::numeric_limits<float>::max();
        bwstats[i].max = -std::numeric_limits<float>::max();
        cgstats[i].min = std::numeric_limits<float>::max();
        cgstats[i].max = -std::numeric_limits<float>::max();
        bgstats[i].min = std::numeric_limits<float>::max();
        bgstats[i].max = -std::numeric_limits<float>::max();
    }
    // compute stats
    for(int i = 0; i < cweights.size(); i++) {
        cwstats[i] = computeStats(cweights[i]);
        bwstats[i] = computeStats(bweights[i]);
        cgstats[i] = computeStats(cgrad[i]);
        bgstats[i] = computeStats(bgrad[i]);
        actstats[i] = computeStats(activations[i]);
    }
    for(int i = 0; i < cweights.size(); i++) {
        stats[i] = {cwstats[i].max, cwstats[i].min, cwstats[i].mean, cwstats[i].std};
    }
    for(int i = 0; i < bweights.size(); i++) {
        stats[cweights.size() + i] = {bwstats[i].max, bwstats[i].min, bwstats[i].mean, bwstats[i].std};
    }
    for(int i = 0; i < cgstats.size(); i++) {
        stats[(cweights.size() * 2) + i] = {cgstats[i].max, cgstats[i].min, cgstats[i].mean, cgstats[i].std};
    }
    for(int i = 0; i < bgstats.size(); i++) {
        stats[(cweights.size() * 3) + i] = {bgstats[i].max, bgstats[i].min, bgstats[i].mean, bgstats[i].std};
    }
    for(int i = 0; i < actstats.size(); i++) {
        stats[(cweights.size() * 4) + i] = {actstats[i].max, actstats[i].min, actstats[i].mean, actstats[i].std};
    }
}


/**
 * @brief compute statistics for MNN2D batch
 * @param cweights coefficients
 * @param bweights biases
 * @param cgrad gradients for coefficients
 * @param bgrad gradients for biases
 * @param stats statistics (maximum, minimum, mean, standard deviation)
 */
void computeStatsForCsv(const std::vector<std::vector<std::vector<float>>> &cweights, 
                        const std::vector<std::vector<std::vector<float>>> &bweights,
                        const std::vector<std::vector<std::vector<float>>> &cgrad,
                        const std::vector<std::vector<std::vector<float>>> &bgrad,
                        const std::vector<std::vector<std::vector<float>>> &activations,
                        std::vector<std::vector<float>> &stats)
{
    std::vector<Statistics> cwstats(cweights.size());
    std::vector<Statistics> bwstats(bweights.size());
    std::vector<Statistics> cgstats(cgrad.size());
    std::vector<Statistics> bgstats(bgrad.size());
    std::vector<Statistics> actstats(activations.size());
    stats.clear();
    stats.resize(cweights.size() * 5, std::vector<float>(4, 0.0f));
    // Compute statistics
    float sum = 0.0f;
    float sum_sq = 0.0f;
    size_t count = 0;
    // set min and max
    for(int i = 0; i < cweights.size(); i++) {
        cwstats[i].min = std::numeric_limits<float>::max();
        cwstats[i].max = -std::numeric_limits<float>::max();
        bwstats[i].min = std::numeric_limits<float>::max();
        bwstats[i].max = -std::numeric_limits<float>::max();
        cgstats[i].min = std::numeric_limits<float>::max();
        cgstats[i].max = -std::numeric_limits<float>::max();
        bgstats[i].min = std::numeric_limits<float>::max();
        bgstats[i].max = -std::numeric_limits<float>::max();
    }
    // compute stats
    for(int i = 0; i < cweights.size(); i++) {
        cwstats[i] = computeStats(cweights[i]);
        bwstats[i] = computeStats(bweights[i]);
        cgstats[i] = computeStats(cgrad[i]);
        bgstats[i] = computeStats(bgrad[i]);
        actstats[i] = computeStats(activations[i]);
    }
    for(int i = 0; i < cweights.size(); i++) {
        stats[i] = {cwstats[i].max, cwstats[i].min, cwstats[i].mean, cwstats[i].std};
    }
    for(int i = 0; i < bweights.size(); i++) {
        stats[cweights.size() + i] = {bwstats[i].max, bwstats[i].min, bwstats[i].mean, bwstats[i].std};
    }
    for(int i = 0; i < cgstats.size(); i++) {
        stats[(cweights.size() * 2) + i] = {cgstats[i].max, cgstats[i].min, cgstats[i].mean, cgstats[i].std};
    }
    for(int i = 0; i < bgstats.size(); i++) {
        stats[(cweights.size() * 3) + i] = {bgstats[i].max, bgstats[i].min, bgstats[i].mean, bgstats[i].std};
    }
    for(int i = 0; i < actstats.size(); i++) {
        stats[(cweights.size() * 4) + i] = {actstats[i].max, actstats[i].min, actstats[i].mean, actstats[i].std};
    }
}