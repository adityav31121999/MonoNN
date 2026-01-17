#include "cppop.hpp"
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <cmath>

/**
 * @brief Calculates the mean of a vector of floats.
 * @param in The input vector.
 * @return The mean value of the elements in the vector.
 */
float mean(const std::vector<float>& in) {
    if (in.empty()) {
        return 0.0f;
    }
    float sum = 0.0f;
    sum = std::accumulate(in.begin(), in.end(), 0.0f);
    return sum / in.size();
}


/**
 * @brief Calculates the sum of the squared differences between two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @return The sum of squared differences.
 */
float sumOfSquareOfDiff(const std::vector<float>& a, const std::vector<float> b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("sumOfSquareOfDiff: Sizes of vector should be same: "
                                 + std::to_string(a.size()) + " vs " + std::to_string(b.size()));
    }

    float sum = 0.0f;
    for(int i = 0; i < a.size(); i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
}


/**
 * @brief Calculates the sum of the squared differences between a vector and a scalar.
 * @param a The input vector.
 * @param b The scalar value.
 * @return The sum of squared differences.
 */
float sumOfSquareOfDiff(const std::vector<float>& a, const float b) {
    float sum = 0.0f;

    for(int i = 0; i < a.size(); i++) {
        sum += (a[i] - b) * (a[i] - b);
    }
    return sum;
}


/**
 * @brief Calculates the Mean Squared Error (MSE) between output and target vectors.
 * @param output The predicted output vector from the network.
 * @param target The ground truth vector.
 * @return The mean squared error.
 */
float mse(const std::vector<float>& output, const std::vector<float>& target) {
    if (output.size() != target.size() || output.empty()) {
        throw std::invalid_argument("Output and target vectors must have the same, non-zero size for MSE.");
    }
    float sum_sq_err = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        float error = clamp(output[i] - target[i]);
        sum_sq_err += error * error;
    }
    return clamp(sum_sq_err / output.size());
}


/**
 * @brief Calculates the derivative of the Mean Squared Error (MSE) loss.
 * @param output The predicted output vector from the network.
 * @param target The ground truth vector.
 * @return A vector representing the gradient of the MSE loss.
 */
std::vector<float> mseDer(const std::vector<float>& output, const std::vector<float>& target) {
    if (output.size() != target.size()) {
        throw std::invalid_argument("Output and target vectors must have the same size for MSE derivative.");
    }
    std::vector<float> derivative(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        derivative[i] = 2.0f * (output[i] - target[i]) / output.size();
        derivative[i] = clamp(derivative[i]);
    }
    return derivative;
}


/**
 * @brief Calculates the Cross-Entropy loss for a single-label classification task.
 * @param output The predicted probability distribution from the network (after softmax).
 * @param target The one-hot encoded ground truth vector.
 * @return The cross-entropy loss.
 */
float crossEntropy(const std::vector<float>& output, const std::vector<float>& target) {
    if (output.size() != target.size() || output.empty()) {
        throw std::invalid_argument("Output and target vectors must have the same, non-zero size for Cross-Entropy.");
    }
    float loss = 0.0f;
    const float epsilon = 1e-9f;       // Small value to prevent log(0)
    for (size_t i = 0; i < output.size(); ++i) {
        // Clamp output to prevent log(0)
        float clamped_output = std::max(clamp(output[i]), epsilon);
        loss += target[i] * std::log(clamped_output);
    }
    return clamp(-loss);
}


/**
 * @brief Calculates the derivative of the Cross-Entropy loss.
 * @param output The predicted probability distribution from the network.
 * @param target The one-hot encoded ground truth vector.
 * @return A vector representing the gradient of the cross-entropy loss.
 */
std::vector<float> crossEntropyDer(const std::vector<float>& output, const std::vector<float>& target) {
    if (output.size() != target.size()) {
        throw std::invalid_argument("Output and target vectors must have the same size for Cross-Entropy derivative.");
    }
    std::vector<float> derivative(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        derivative[i] = -target[i] / (output[i] + 1e-9f);
        derivative[i] = clamp(derivative[i]);
    }
    return derivative;
}


/**
 * @brief Calculates the Binary Cross-Entropy loss, typically used for binary classification.
 * @param output The predicted output vector (probabilities, usually after a sigmoid).
 * @param target The binary ground truth vector (0s and 1s).
 * @return The binary cross-entropy loss.
 */
float binaryCrossEntropy(const std::vector<float>& output, const std::vector<float>& target) {
    if (output.size() != target.size() || output.empty()) {
        throw std::invalid_argument("Output and target vectors must have the same, non-zero size for Binary Cross-Entropy.");
    }
    float loss = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        // Add a small epsilon to prevent log(0) or log(1) for 0 or 1
        loss += target[i] * std::log(clamp(output[i]) + 1e-9f) + (1.0f - target[i]) * std::log(1.0f - clamp(output[i]) + 1e-9f);
    }
    return clamp(-loss / output.size());
}


/**
 * @brief Calculates the derivative of the Binary Cross-Entropy loss.
 * @param output The predicted output vector.
 * @param target The binary ground truth vector.
 * @return A vector representing the gradient of the binary cross-entropy loss.
 */
std::vector<float> binaryCrossEntropyDer(const std::vector<float>& output, const std::vector<float>& target) {
    if (output.size() != target.size()) {
        throw std::invalid_argument("Output and target vectors must have the same size for Binary Cross-Entropy derivative.");
    }
    std::vector<float> derivative(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        // Derivative with respect to the output
        derivative[i] = -(target[i] / (output[i] + 1e-9f) - (1.0f - target[i]) / (1.0f - output[i] + 1e-9f)) / output.size();
        derivative[i] = clamp(derivative[i]);
    }
    return derivative;
}


/**
 * @brief Calculates the Categorical Cross-Entropy loss for a batch of outputs.
 * @param output A 2D vector of predicted probability distributions for a batch.
 * @param target A 2D vector of one-hot encoded ground truth vectors for the batch.
 * @return The average categorical cross-entropy loss over the batch.
 */
float categoricalCrossEntropy(const std::vector<std::vector<float>>& output, const std::vector<std::vector<float>>& target) {
    if (output.size() != target.size() || output[0].size() != target[0].size()) {
        throw std::runtime_error("Dimension mismatch: " + std::to_string(output.size()) + "x" + std::to_string(output[0].size()) 
                                                        + " vs " + std::to_string(target.size()) + "x" + std::to_string(target[0].size()));
    }
    float loss = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        for (size_t j = 0; j < output[i].size(); ++j) {
            loss += target[i][j] * std::log(clamp(output[i][j]) + 1e-9f);
        }
    }
    return clamp(-loss / output.size());
}


/**
 * @brief Calculates the derivative of the Categorical Cross-Entropy loss for a batch.
 * @param output A 2D vector of predicted probability distributions for a batch.
 * @param target A 2D vector of one-hot encoded ground truth vectors for the batch.
 * @return A 2D vector representing the gradient of the loss for the batch.
 */
std::vector<std::vector<float>> categoricalCrossEntropyDer(const std::vector<std::vector<float>>& output, const std::vector<std::vector<float>>& target) {
    if (output.size() != target.size() || output[0].size() != target[0].size()) {
        throw std::runtime_error("Dimension mismatch: " + std::to_string(output.size()) + "x" + std::to_string(output[0].size()) 
                                                        + " vs " + std::to_string(target.size()) + "x" + std::to_string(target[0].size()));
    }
    std::vector<std::vector<float>> derivative(output.size(), std::vector<float>(output[0].size()));
    for (size_t i = 0; i < output.size(); ++i) {
        for (size_t j = 0; j < output[i].size(); ++j) {
            derivative[i][j] = -target[i][j] / (output[i][j] + 1e-9f) / output.size();
            derivative[i][j] = clamp(derivative[i][j]);
        }
    }
    return derivative;
}