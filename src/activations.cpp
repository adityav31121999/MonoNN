#include "include/activations.hpp"
#include <vector>
#include <cmath>

/// sigmoid activation function and its derivative

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float sigmoidDer(float x) {
    float sig = sigmoid(x);
    return sig * (1.0f - sig);
}

std::vector<float> sigmoid(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sigmoid(x[i]);
    }
    return result;
}

std::vector<float> sigmoidDer(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sigmoidDer(x[i]);
    }
    return result;
}

std::vector<std::vector<float>> sigmoid(const std::vector<std::vector<float>>& x) {
    std::vector<std::vector<float>> result(x.size(), std::vector<float>(x[0].size()));
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < x[i].size(); ++j) {
            result[i][j] = sigmoid(x[i][j]);
        }
    }
    return result;
}

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

float relu(float x) {
    return x > 0 ? x : 0;
}

float reluDer(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

std::vector<float> relu(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = relu(x[i]);
    }
    return result;
}

std::vector<float> reluDer(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = reluDer(x[i]);
    }
    return result;
}

std::vector<std::vector<float>> relu(const std::vector<std::vector<float>>& x) {
    std::vector<std::vector<float>> result(x.size(), std::vector<float>(x[0].size()));
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < x[i].size(); ++j) {
            result[i][j] = relu(x[i][j]);
        }
    }
    return result;
}

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

std::vector<float> softmax(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    float sumExp = 0.0f;
    for (float val : x) {
        sumExp += std::exp(val);
    }
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i]) / sumExp;
    }
    return result;
}

std::vector<float> softmaxDer(const std::vector<float>& x) {
    std::vector<float> s = softmax(x);
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = s[i] * (1.0f - s[i]);
    }
    return result;
}

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

std::vector<float> softmaxDer(const std::vector<float>& x, float temp) {
    std::vector<float> s = softmax(x, temp);
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = (s[i] * (1.0f - s[i])) / temp;
    }
    return result;
}
