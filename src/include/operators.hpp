#ifndef OPERATORS_HPP
#define OPERATORS_HPP 1
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <cmath>
#include <string>
#include <map>
#include <functional>
#include "progress.hpp"

#include <thread>
#include <mutex>

// Helper to determine thread count
inline unsigned int get_thread_count(size_t work_size) {
    unsigned int num = std::thread::hardware_concurrency();
    return num == 0 ? 2 : std::min<unsigned int>(num, static_cast<unsigned int>(work_size));
}

void printClassDistribution(const std::vector<std::filesystem::path>& filePaths, int outSize);

// file operations for weights serialization

void createBinFile(const std::string& fileAddress, unsigned long long param);
void serializeWeights(const std::vector<std::vector<std::vector<float>>>& cweights,
                        const std::vector<std::vector<std::vector<float>>>& bweights,
                        const std::string& fileAddress);
void deserializeWeights(std::vector<float>& cweights, std::vector<float>& bweights,
                        const std::string& fileAddress);
void deserializeWeights(std::vector<std::vector<std::vector<float>>>& cweights,
                        std::vector<std::vector<std::vector<float>>>& bweights,
                        const std::vector<int>& width, const std::vector<int>& height,
                        const std::string& fileAddress);

// image access and manipulation
#define NOMINMAX
#include <opencv2/core.hpp>

std::vector<std::vector<float>> cvMat2vec(const cv::Mat& mat);
cv::Mat vec2cvMat(const std::vector<std::vector<float>>& vec);
cv::Mat image2grey(const std::string& path2image);
std::vector<std::vector<float>> image2matrix(const std::string& path2image, bool isGreyOrRGB);
std::vector<std::vector<std::vector<float>>> image2channels(const std::string& path2image);

#endif // OPERATORS_HPP