#include "operators.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

/// sigmoid activation function and its derivative

/**
 * @brief Computes the sigmoid activation for a single float value.
 * @param x The input value.
 * @return The sigmoid of x.
 */
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/**
 * @brief Computes the derivative of the sigmoid function for a single float value.
 * @param x The input value.
 * @return The derivative of the sigmoid at x.
 */
float sigmoidDer(float x) {
    float sig = sigmoid(x);
    return sig * (1.0f - sig);
}

/**
 * @brief Applies the sigmoid activation function element-wise to a vector.
 * @param x The input vector.
 * @return A new vector with the sigmoid function applied to each element.
 */
std::vector<float> sigmoid(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sigmoid(x[i]);
    }
    return result;
}

/**
 * @brief Computes the derivative of the sigmoid function element-wise for a vector.
 * @param x The input vector.
 * @return A new vector with the sigmoid derivative applied to each element.
 */
std::vector<float> sigmoidDer(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sigmoidDer(x[i]);
    }
    return result;
}

/**
 * @brief Applies the sigmoid activation function element-wise to a 2D vector (matrix).
 * @param x The input matrix.
 * @return A new matrix with the sigmoid function applied to each element.
 */
std::vector<std::vector<float>> sigmoid(const std::vector<std::vector<float>>& x) {
    std::vector<std::vector<float>> result(x.size(), std::vector<float>(x[0].size()));
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < x[i].size(); ++j) {
            result[i][j] = sigmoid(x[i][j]);
        }
    }
    return result;
}

/**
 * @brief Computes the derivative of the sigmoid function element-wise for a 2D vector (matrix).
 * @param x The input matrix.
 * @return A new matrix with the sigmoid derivative applied to each element.
 */
std::vector<std::vector<float>> sigmoidDer(const std::vector<std::vector<float>>& x) {
    std::vector<std::vector<float>> result(x.size(), std::vector<float>(x[0].size()));
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < x[i].size(); ++j) {
            result[i][j] = sigmoidDer(x[i][j]);
        }
    }
    return result;
}

/// ReLU activation function and its derivative

/**
 * @brief Computes the Rectified Linear Unit (ReLU) activation for a single float value.
 * @param x The input value.
 * @return x if x > 0, otherwise 0.
 */
float relu(float x) {
    return x > 0 ? x : 0;
}

/**
 * @brief Computes the derivative of the ReLU function for a single float value.
 * @param x The input value.
 * @return 1.0f if x > 0, otherwise 0.0f.
 */
float reluDer(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

/**
 * @brief Applies the ReLU activation function element-wise to a vector.
 * @param x The input vector.
 * @return A new vector with the ReLU function applied to each element.
 */
std::vector<float> relu(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = relu(x[i]);
    }
    return result;
}

/**
 * @brief Computes the derivative of the ReLU function element-wise for a vector.
 * @param x The input vector.
 * @return A new vector with the ReLU derivative applied to each element.
 */
std::vector<float> reluDer(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = reluDer(x[i]);
    }
    return result;
}

/**
 * @brief Applies the ReLU activation function element-wise to a 2D vector (matrix).
 * @param x The input matrix.
 * @return A new matrix with the ReLU function applied to each element.
 */
std::vector<std::vector<float>> relu(const std::vector<std::vector<float>>& x) {
    std::vector<std::vector<float>> result(x.size(), std::vector<float>(x[0].size()));
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < x[i].size(); ++j) {
            result[i][j] = relu(x[i][j]);
        }
    }
    return result;
}

/**
 * @brief Computes the derivative of the ReLU function element-wise for a 2D vector (matrix).
 * @param x The input matrix.
 * @return A new matrix with the ReLU derivative applied to each element.
 */
std::vector<std::vector<float>> reluDer(const std::vector<std::vector<float>>& x) {
    std::vector<std::vector<float>> result(x.size(), std::vector<float>(x[0].size()));
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < x[i].size(); ++j) {
            result[i][j] = reluDer(x[i][j]);
        }
    }
    return result;
}

/// softmax activation function and its derivative

/**
 * @brief Computes the softmax activation for a vector, converting scores to probabilities.
 * @param x The input vector of scores.
 * @return A vector of probabilities that sum to 1.
 */
std::vector<float> softmax(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    
    // Find max value for numerical stability
    float max_val = *std::max_element(x.begin(), x.end());
    
    float sumExp = 0.0f;
    for (float val : x) {
        sumExp += std::exp(val - max_val);
    }
    
    // Prevent division by zero
    if (sumExp == 0.0f || std::isnan(sumExp) || std::isinf(sumExp)) {
        // Return uniform distribution
        float uniform_val = 1.0f / x.size();
        std::fill(result.begin(), result.end(), uniform_val);
        return result;
    }
    
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i] - max_val) / sumExp;
        // Additional safety check
        if (std::isnan(result[i]) || std::isinf(result[i])) {
            result[i] = 1.0f / x.size();
        }
    }
    return result;
}

/**
 * @brief Computes the derivative of the softmax function for a vector.
 * @param x The input vector.
 * @return A vector representing the derivative of softmax.
 */
std::vector<float> softmaxDer(const std::vector<float>& x) {
    std::vector<float> s = softmax(x);
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = s[i] * (1.0f - s[i]);
    }
    return result;
}

/**
 * @brief Computes the softmax activation with a temperature parameter.
 * @param x The input vector of scores.
 * @param temp The temperature parameter. Higher values result in a softer probability distribution.
 * @return A vector of probabilities.
 */
std::vector<float> softmax(const std::vector<float>& x, float temp) {
    std::vector<float> result(x.size());
    float sumExp = 0.0f;
    for (float val : x) {
        sumExp += std::exp(val / temp);
    }
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i] / temp) / sumExp;
    }
    return result;
}

/**
 * @brief Computes the derivative of the softmax function with a temperature parameter.
 * @param x The input vector.
 * @param temp The temperature parameter.
 * @return A vector representing the derivative of softmax with temperature.
 */
std::vector<float> softmaxDer(const std::vector<float>& x, float temp) {
    std::vector<float> s = softmax(x, temp);
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = (s[i] * (1.0f - s[i])) / temp;
    }
    return result;
}

//----------------Least of them all (LOTA) and it's derivative----------------//

/**
 * @brief Applies the LOTA activation function to a vector.
 *        LOTA(x_i) = (x_i + abs(min(x))) / sum(x_j + abs(min(x)))
 * @param y Input vector (const reference).
 * @return A new vector containing the LOTA results.
 */
std::vector<float> LOTA(const std::vector<float>& y) {
    if (y.empty()) {
        return {}; // Return empty for empty input
    }
    if (y.size() == 1) {
        return {1.0f}; // Single element always results in probability 1
    }

    // Find the minimum value in the input vector
    float min_val = *std::min_element(y.begin(), y.end());
    float abs_min_val = std::abs(min_val);

    // Create a temporary vector for transformed values (x_i + abs(min_val))
    std::vector<float> transformed_x = y; // Copy constructor
    float sum = 0.0f;
    for(float& val : transformed_x) {
        val += abs_min_val;
        sum += val;
    }

    // Normalize the transformed vector
    if (sum > 0.0f) { // Avoid division by zero
        for(float& val : transformed_x) {
            val /= sum;
        }
    } else if (!transformed_x.empty()) {
        // Handle sum == 0 case (e.g., all elements were -abs_min_val) -> uniform distribution
        float uniform_prob = 1.0f / static_cast<float>(transformed_x.size());
        std::fill(transformed_x.begin(), transformed_x.end(), uniform_prob);
    }
    // If sum is non-positive and vector is empty, it returns empty anyway

    return transformed_x; // Return the normalized vector
}


/**
 * @brief Calculates the derivative of the LOTA activation function for a vector.
 *        LOTA'(x_i) = (sum - transformed_x_i) / sum^2
 *        where transformed_x_i = x_i + abs(min(x)) and sum = sum(transformed_x_j)
 * @param y Input vector (const reference).
 * @return A new vector containing the LOTA derivative results.
 */
std::vector<float> LOTAder(const std::vector<float>& y) {
     if (y.empty()) {
        return {};
    }
     if (y.size() == 1) {
         // Derivative for a single element LOTA(x)=1 is 0
         return {0.0f};
     }

    // Find the minimum value
    float min_val = *std::min_element(y.begin(), y.end());
    float abs_min_val = std::abs(min_val);

    // Calculate transformed values and their sum
    std::vector<float> transformed_x = y; // Copy
    float sum = 0.0f;
    for(float& val : transformed_x) {
        val += abs_min_val;
        sum += val;
    }

    // Calculate the derivative
    std::vector<float> derivative_x(y.size());
    float sum_sq = sum * sum; // Calculate sum squared once

    if (sum > 0.0f) { // Avoid division by zero
        for (size_t i = 0; i < y.size(); ++i) {
            // Derivative: (sum - transformed_element) / sum^2
            derivative_x[i] = (sum - transformed_x[i]) / sum_sq;
        }
    } else {
        // Handle sum == 0 case (derivative is likely 0 or undefined)
        std::fill(derivative_x.begin(), derivative_x.end(), 0.0f);
    }

    return derivative_x;
}


/**
 * @brief Applies the LOTA activation function to a 2D vector (matrix),
 *        considering only relevant elements defined by 't' and 'attentionType'.
 * @param y Input 2D vector (const reference).
 * @param t Dimension limit (passed by value).
 * @param attentionType If true, process only the lower triangle (incl. diagonal); otherwise, process up to t x t square (passed by value).
 * @return A new 2D vector with LOTA applied to relevant elements, others potentially zeroed.
 */
std::vector<std::vector<float>> LOTA(const std::vector<std::vector<float>>& y, int t, bool attentionType) { // Pass t and attentionType by value
    if (y.empty() || y[0].empty() || t <= 0) return {{}}; // Handle edge cases

    std::vector<std::vector<float>> x = y; // Operate on a copy

    // Handle 1x1 case explicitly if t=1
    if (t == 1 && !x.empty() && !x[0].empty()) {
        x[0][0] = 1.0f;
        // Zero out other elements if needed based on desired output shape
        // for (size_t j = 1; j < x[0].size(); ++j) x[0][j] = 0.0f;
        // for (size_t i = 1; i < x.size(); ++i) std::fill(x[i].begin(), x[i].end(), 0.0f);
        return x;
    }

    // Find the minimum value in the relevant region
    float min_val = (std::numeric_limits<float>::max)(); // Parenthesize to avoid macro
    bool found_value = false;
    size_t max_rows = (std::min)((size_t)t, x.size()); // Parenthesize to avoid macro
    for (size_t i = 0; i < max_rows; ++i) {
        size_t limit_j = attentionType ? (i + 1) : (size_t)t;
        limit_j = (std::min)(limit_j, x[i].size()); // Boundary check cols, parenthesize
        for (size_t j = 0; j < limit_j; ++j) {
            min_val = (std::min)(min_val, x[i][j]); // Parenthesize to avoid macro
            found_value = true;
        }
    }
    if (!found_value) 
        min_val = 0.0f; // Handle case where relevant region is effectively empty

    float abs_min_val = std::abs(min_val);

    // Transform relevant elements: element + abs(min_val) and calculate sum
    float sum = 0.0f;
    int relevant_count = 0;
    for (size_t i = 0; i < max_rows; ++i) {
        size_t limit_j = attentionType ? (i + 1) : (size_t)t;
        limit_j = (std::min)(limit_j, x[i].size()); // Boundary check cols, parenthesize
        for (size_t j = 0; j < limit_j; ++j) {
            x[i][j] += abs_min_val;
            sum += x[i][j];
            relevant_count++;
        }
        // Zero out non-relevant elements in the row if attentionType is true
        if (attentionType) {
            for (size_t j = limit_j; j < x[i].size(); ++j) {
                x[i][j] = 0.0f;
            }
        }
    }
    // Normalize relevant elements by the global sum
    if (sum > 0.0f) {
        for (size_t i = 0; i < max_rows; ++i) {
            size_t limit_j = attentionType ? (i + 1) : (size_t)t;
            limit_j = (std::min)(limit_j, x[i].size()); // Boundary check cols, parenthesize
            for (size_t j = 0; j < limit_j; ++j) {
                x[i][j] /= sum;
            }
        }
    } 
    else if (relevant_count > 0) {
        // Handle sum=0 case -> uniform distribution over relevant elements
        float uniform_val = 1.0f / static_cast<float>(relevant_count);
        for (size_t i = 0; i < max_rows; ++i) {
            size_t limit_j = attentionType ? (i + 1) : (size_t)t;
            limit_j = (std::min)(limit_j, x[i].size()); // Boundary check cols, parenthesize
            for (size_t j = 0; j < limit_j; ++j) {
                x[i][j] = uniform_val;
            }
        }
    }
    // Non-relevant elements remain 0 (if zeroed out) or their original value
    return x; // Return the modified 2D vector
}


/**
 * @brief Derivative of the LOTA activation function for a 2D vector (matrix),
 *        considering only relevant elements defined by 't' and 'attentionType'.
 * @param y Input 2D vector (const reference).
 * @param t Dimension limit (passed by value).
 * @param attentionType If true, process only the lower triangle; otherwise, up to t x t square (passed by value).
 * @return A new 2D vector with LOTA derivative applied to relevant elements.
 */
std::vector<std::vector<float>> LOTAder(const std::vector<std::vector<float>>& y, int t, bool attentionType) { // Pass t and attentionType by value
    if (y.empty() || y[0].empty() || t <= 0) return {{}};

    std::vector<std::vector<float>> result = y; // Create a copy to store derivatives

    // Handle 1x1 case explicitly if t=1
    if (t == 1 && !result.empty() && !result[0].empty()) {
        result[0][0] = 0.0f; // Derivative of LOTA(x)=1 is 0
        return result;
    }

    // Find the minimum value in the relevant region
    float min_val = (std::numeric_limits<float>::max)(); // Parenthesize to avoid macro
    bool found_value = false;
    size_t max_rows = (std::min)((size_t)t, y.size()); // Use y for reading min, parenthesize
    for (size_t i = 0; i < max_rows; ++i) {
        size_t limit_j = attentionType ? (i + 1) : (size_t)t;
        limit_j = (std::min)(limit_j, y[i].size()); // Boundary check cols, parenthesize
        for (size_t j = 0; j < limit_j; ++j) {
            min_val = (std::min)(min_val, y[i][j]); // Parenthesize to avoid macro
            found_value = true;
        }
    }
    if (!found_value) min_val = 0.0f;

    float abs_min_val = std::abs(min_val);
    // Calculate the sum of (element + abs(min_val)) in the relevant region
    // Also store transformed values temporarily
    float sum = 0.0f;
    std::vector<std::vector<float>> transformed_x = y; // Temp storage
    for (size_t i = 0; i < max_rows; ++i) {
        size_t limit_j = attentionType ? (i + 1) : (size_t)t;
        limit_j = (std::min)(limit_j, y[i].size()); // Boundary check cols, parenthesize
        for (size_t j = 0; j < limit_j; ++j) {
            transformed_x[i][j] = y[i][j] + abs_min_val; // Use original y here
            sum += transformed_x[i][j];
        }
    }

    // Calculate the derivative for each element in the relevant region
    float sum_sq = sum * sum;
    if (sum > 0.0f) { // Avoid division by zero
        for (size_t i = 0; i < max_rows; ++i) {
            size_t limit_j = attentionType ? (i + 1) : (size_t)t;
            limit_j = (std::min)(limit_j, y[i].size()); // Boundary check cols, parenthesize
            for (size_t j = 0; j < limit_j; ++j) {
                // Derivative: (sum - transformed_element) / sum^2
                result[i][j] = (sum - transformed_x[i][j]) / sum_sq;
            }
            // Zero out non-relevant elements in the row if attentionType is true
            if (attentionType) {
                for (size_t j = limit_j; j < result[i].size(); ++j) { // Use result here
                    result[i][j] = 0.0f; // Parenthesize to avoid macro
                }
            }
        }
    } 
    else {
        // Handle sum=0 case (derivative is likely 0 or undefined)
        for (size_t i = 0; i < max_rows; ++i) {
            size_t limit_j = attentionType ? (i + 1) : (size_t)t;
            limit_j = (std::min)(limit_j, y[i].size()); // Boundary check cols, parenthesize
            for (size_t j = 0; j < limit_j; ++j) {
                result[i][j] = 0.0f; // Set derivative to 0
            }
            // Zero out non-relevant elements if attentionType is true
            if (attentionType) {
                for (size_t j = limit_j; j < result[i].size(); ++j) {
                    result[i][j] = 0.0f;
                }
            }
        }
    }

    return result; // Return the derivative matrix
}

std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& x) {
    if (x.empty() || x[0].empty()) return {};
    int rows = x.size();
    int cols = x[0].size();
    std::vector<float> flat_x = flatten(x);
    std::vector<float> flat_softmax = softmax(flat_x);
    return reshape(flat_softmax, rows, cols);
}

std::vector<std::vector<float>> softmaxDer(const std::vector<std::vector<float>>& x) {
    if (x.empty() || x[0].empty()) return {};
    int rows = x.size();
    int cols = x[0].size();
    std::vector<float> flat_x = flatten(x);
    std::vector<float> flat_softmax_der = softmaxDer(flat_x);
    return reshape(flat_softmax_der, rows, cols);
}

std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& x, float temp) {
    if (x.empty() || x[0].empty()) return {};
    int rows = x.size();
    int cols = x[0].size();
    std::vector<float> flat_x = flatten(x);
    std::vector<float> flat_softmax = softmax(flat_x, temp);
    return reshape(flat_softmax, rows, cols);
}

std::vector<std::vector<float>> softmaxDer(const std::vector<std::vector<float>>& x, float temp) {
    if (x.empty() || x[0].empty()) return {};
    int rows = x.size();
    int cols = x[0].size();
    std::vector<float> flat_x = flatten(x);
    std::vector<float> flat_softmax_der = softmaxDer(flat_x, temp);
    return reshape(flat_softmax_der, rows, cols);
}
