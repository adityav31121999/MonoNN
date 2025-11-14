#include "operators.hpp"
#include <stdexcept>
#include <numeric>
#include <cmath>

float mse(const std::vector<float>& output, const std::vector<float>& target) {
    if (output.size() != target.size() || output.empty()) {
        throw std::invalid_argument("Output and target vectors must have the same, non-zero size for MSE.");
    }
    float sum_sq_err = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        float error = output[i] - target[i];
        sum_sq_err += error * error;
    }
    return sum_sq_err / output.size();
}

std::vector<float> mseDer(const std::vector<float>& output, const std::vector<float>& target) {
    if (output.size() != target.size()) {
        throw std::invalid_argument("Output and target vectors must have the same size for MSE derivative.");
    }
    std::vector<float> derivative(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        derivative[i] = 2.0f * (output[i] - target[i]) / output.size();
    }
    return derivative;
}

float crossEntropy(const std::vector<float>& output, const std::vector<float>& target) {
    if (output.size() != target.size() || output.empty()) {
        throw std::invalid_argument("Output and target vectors must have the same, non-zero size for Cross-Entropy.");
    }
    float loss = 0.0f;
    const float epsilon = 1e-7f;  // Increased epsilon for better stability
    for (size_t i = 0; i < output.size(); ++i) {
        // Clamp output to prevent log(0) and log(1)
        float clamped_output = std::max<float>(epsilon, std::min<float>(1.0f - epsilon, output[i]));
        loss += target[i] * std::log(clamped_output);
    }
    return -loss;
}

std::vector<float> crossEntropyDer(const std::vector<float>& output, const std::vector<float>& target) {
    if (output.size() != target.size()) {
        throw std::invalid_argument("Output and target vectors must have the same size for Cross-Entropy derivative.");
    }
    std::vector<float> derivative(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        derivative[i] = -target[i] / (output[i] + 1e-9f);
    }
    return derivative;
}

float binaryCrossEntropy(const std::vector<float>& output, const std::vector<float>& target) {
    if (output.size() != target.size() || output.empty()) {
        throw std::invalid_argument("Output and target vectors must have the same, non-zero size for Binary Cross-Entropy.");
    }
    float loss = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        // Add a small epsilon to prevent log(0) or log(1) for 0 or 1
        loss += target[i] * std::log(output[i] + 1e-9f) + (1.0f - target[i]) * std::log(1.0f - output[i] + 1e-9f);
    }
    return -loss / output.size();
}

std::vector<float> binaryCrossEntropyDer(const std::vector<float>& output, const std::vector<float>& target) {
    if (output.size() != target.size()) {
        throw std::invalid_argument("Output and target vectors must have the same size for Binary Cross-Entropy derivative.");
    }
    std::vector<float> derivative(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        // Derivative with respect to the output
        derivative[i] = -(target[i] / (output[i] + 1e-9f) - (1.0f - target[i]) / (1.0f - output[i] + 1e-9f)) / output.size();
    }
    return derivative;
}

float categoricalCrossEntropy(const std::vector<std::vector<float>>& output, const std::vector<std::vector<float>>& target) {
    if (output.size() != target.size() || output[0].size() != target[0].size()) {
        throw std::runtime_error("Dimension mismatch: " + std::to_string(output.size()) + "x" + std::to_string(output[0].size()) 
                                                        + " vs " + std::to_string(target.size()) + "x" + std::to_string(target[0].size()));
    }
    float loss = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        for (size_t j = 0; j < output[i].size(); ++j) {
            loss += target[i][j] * std::log(output[i][j] + 1e-9f);
        }
    }
    return -loss / output.size();
}
