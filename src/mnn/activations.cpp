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
    float sumExp = 0.0f;
    sumExp = std::accumulate(x.begin(), x.end(), 0.0f, [](float acc, float val) {
        return acc + std::exp(val);
    });
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i]) / sumExp;
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
