#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP 1

#include <vector>
#include <cmath>

float sigmoid(float x);
float sigmoidDer(float x);
std::vector<float> sigmoid(const std::vector<float>& x);
std::vector<float> sigmoidDer(const std::vector<float>& x);
std::vector<std::vector<float>> sigmoid(const std::vector<std::vector<float>>& x);
std::vector<std::vector<float>> sigmoidDer(const std::vector<std::vector<float>>& x);

float relu(float x);
float reluDer(float x);
std::vector<float> relu(const std::vector<float>& x);
std::vector<float> reluDer(const std::vector<float>& x);
std::vector<std::vector<float>> relu(const std::vector<std::vector<float>>& x);
std::vector<std::vector<float>> reluDer(const std::vector<std::vector<float>>& x);

std::vector<float> softmax(const std::vector<float>& x);
std::vector<float> softmaxDer(const std::vector<float>& x);
std::vector<float> softmax(const std::vector<float>& x, float temp);
std::vector<float> softmaxDer(const std::vector<float>& x, float temp);

#endif // ACTIVATIONS_HPP