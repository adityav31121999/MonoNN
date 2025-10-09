#ifndef LOSS_HPP
#define LOSS_HPP 1

#include <vector>
#include <cmath>

float mse(const std::vector<float>& output, const std::vector<float>& target);
std::vector<float> mseDer(const std::vector<float>& output, const std::vector<float>& target);

float crossEntropy(const std::vector<float>& output, const std::vector<float>& target);
std::vector<float> crossEntropyDer(const std::vector<float>& output, const std::vector<float>& target);

float binaryCrossEntropy(const std::vector<float>& output, const std::vector<float>& target);
std::vector<float> binaryCrossEntropyDer(const std::vector<float>& output, const std::vector<float>& target);

#endif // LOSS_HPP