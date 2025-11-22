#include "mnn.hpp"
#include "mnn2d.hpp"
#include <opencv2/core.hpp>      // For cv::Mat
#include <opencv2/imgcodecs.hpp> // For cv::imread
#include <opencv2/imgproc.hpp>   // For cv::cvtColor, cv::split
#include <stdexcept>

/**
 * @brief cvMat to 2d vector
 * @param mat cvMat input
 * @return 2d vector
 */
std::vector<std::vector<float>> cvMat2vec(const cv::Mat& mat) {
    if (mat.empty()) {
        return {};
    }

    // Ensure the matrix is of a floating-point type for consistency
    cv::Mat temp;
    if (mat.type() != CV_32F) {
        mat.convertTo(temp, CV_32F);
    } else {
        temp = mat;
    }

    std::vector<std::vector<float>> vec(temp.rows);
    for (int i = 0; i < temp.rows; ++i) {
        // Assign the row data directly
        vec[i].assign((float*)temp.ptr(i), (float*)temp.ptr(i) + temp.cols);
    }
    return vec;
}

/**
 * @brief 2d vector to cvMat
 * @param vec 2d vector input
 * @return cvMat
 */
cv::Mat vec2cvMat(const std::vector<std::vector<float>>& vec) {
    if (vec.empty() || vec[0].empty()) {
        return cv::Mat();
    }

    int rows = vec.size();
    int cols = vec[0].size();

    // Create a CV_32F matrix
    cv::Mat mat(rows, cols, CV_32F);

    for (int i = 0; i < rows; ++i) {
        if (vec[i].size() != static_cast<size_t>(cols)) {
            throw std::invalid_argument("Input vector has inconsistent column sizes.");
        }
        // Copy the data from the vector row to the matrix row
        memcpy(mat.ptr<float>(i), vec[i].data(), cols * sizeof(float));
    }
    return mat;
}

/**
 * @brief convert single image into grey channel
 * @param path2image image location
 * @return cvMat grayscale image
 */
cv::Mat image2grey(const std::string& path2image) {
    cv::Mat image = cv::imread(path2image, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Could not open or find the image: " + path2image);
    }
    cv::Mat greyMat;
    cv::cvtColor(image, greyMat, cv::COLOR_BGR2GRAY);
    return greyMat;
}

/**
 * @brief convert single image to R, G, B and Grey channels
 * @param path2image image location
 * @return 3d vector of shape (4, height, width) containing B, G, R, and Grey channels
 */
std::vector<std::vector<std::vector<float>>> image2channels(const std::string& path2image) {
    cv::Mat image = cv::imread(path2image, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Could not open or find the image: " + path2image);
    }

    std::vector<cv::Mat> bgr_channels;
    cv::split(image, bgr_channels); // Splits into B, G, R

    cv::Mat greyMat;
    cv::cvtColor(image, greyMat, cv::COLOR_BGR2GRAY);

    std::vector<std::vector<std::vector<float>>> result;
    result.push_back(cvMat2vec(bgr_channels[0])); // Blue
    result.push_back(cvMat2vec(bgr_channels[1])); // Green
    result.push_back(cvMat2vec(bgr_channels[2])); // Red
    result.push_back(cvMat2vec(greyMat));         // Grayscale

    return result;
}