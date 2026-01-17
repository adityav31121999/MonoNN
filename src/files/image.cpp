#include "mnn1d.hpp"
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
 * @brief Convert single image to either grayscale matrix or flattened RGB matrix (row-wise)
 * @param path2image Full path to the image file
 * @param isRGB false (0): return grayscale 2D matrix (height x width)
 * true  (1): return color 2D matrix (height x (width * 3)) with B,G,R concatenated per row
 * @return 2D vector of floats representing the image
 */
std::vector<std::vector<float>> image2matrix(const std::string& path2image, bool isRGB) {
    if (!isRGB) {
        // Grayscale mode
        return cvMat2vec(image2grey(path2image));
    }
    else {
        // Color mode
        cv::Mat image = cv::imread(path2image, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Could not open or find the image: " + path2image);
        }

        int height = image.rows;
        int width = image.cols;

        // Convert to float
        cv::Mat float_img;
        image.convertTo(float_img, CV_32F);

        // Split into B, G, R channels (OpenCV order: BGR)
        std::vector<cv::Mat> bgr(3);
        cv::split(float_img, bgr);  // bgr[0]=Blue, bgr[1]=Green, bgr[2]=Red

        // Create result: height rows, each with width*3 columns: B G R B G R ...
        std::vector<std::vector<float>> result(height, std::vector<float>(width * 3));

        for (int i = 0; i < height; ++i) {
            float* row_ptr = result[i].data();
            const float* b_row = bgr[0].ptr<float>(i);
            const float* g_row = bgr[1].ptr<float>(i);
            const float* r_row = bgr[2].ptr<float>(i);

            for (int j = 0; j < width; ++j) {
                row_ptr[3 * j + 0] = b_row[j];  // Blue
                row_ptr[3 * j + 1] = g_row[j];  // Green
                row_ptr[3 * j + 2] = r_row[j];  // Red
            }
        }

        return result;
    }
}

/**
 * @brief convert single image to R, G, B and Grey channels
 * @param path2image image location
 * @param isRGB is image RGB 1, else 0 for grey image
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