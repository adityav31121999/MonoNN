#include "mnn.hpp"
#include <vector>
#include <stdexcept>
#include <thread>
#include <algorithm>
#include <iostream>

/**
 * @brief Overloaded operator for vector addition.
 * @param a Input vector A.
 * @param b Input vector B.
 * @return Resulting vector after addition.
 * @throws std::invalid_argument if dimensions are incompatible.
 */
std::vector<float> operator+(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Incompatible dimensions for addition");
    }

    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] + b[i];
    }
    return result;
}


/**
 * @brief Overloaded operator for matrix addition.
 * @param a Input matrix A.
 * @param b Input matrix B.
 * @return Resulting matrix after addition.
 * @throws std::invalid_argument if dimensions are incompatible.
 */
std::vector<std::vector<float>> operator+(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        throw std::invalid_argument("Incompatible dimensions for addition");
    }

    std::vector<std::vector<float>> result(a.size(), std::vector<float>(a[0].size()));
    for (size_t i = 0; i < a.size(); i++) {
        for (size_t j = 0; j < a[0].size(); j++) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}


/**
 * @brief Overloaded operator for vector-matrix multiplication.
 * @param a Input vector.
 * @param b Input matrix.
 * @return Resulting vector after multiplication.
 * @throws std::invalid_argument if dimensions are incompatible.
 */
std::vector<float> operator*(const std::vector<float>& a, const std::vector<std::vector<float>>& b) {
    if(a.size() != b.size()) {
        throw std::invalid_argument("Incompatible dimensions for multiplication");
    }
    // vector to store result
    std::vector<float> result(b[0].size(), 0.0f);
    for(size_t j = 0; j < b[0].size(); j++) {
        for(size_t i = 0; i < a.size(); i++) {
            result[j] += a[i] * b[i][j];
        }
    }

    return result;
}


/**
 * @brief Overloaded operator for matrix-matrix multiplication.
 * @param a Input matrix A.
 * @param b Input matrix B.
 * @return Resulting matrix after multiplication.
 * @throws std::invalid_argument if dimensions are incompatible.
 */
std::vector<std::vector<float>> operator*(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    if(a[0].size() != b.size()) {
        throw std::invalid_argument("Incompatible dimensions for multiplication");
    }
    // matrix to store result
    std::vector<std::vector<float>> result(a.size(), std::vector<float>(b[0].size(), 0.0f));
    for(size_t i = 0; i < a.size(); i++) {
        for(size_t j = 0; j < b[0].size(); j++) {
            for(size_t k = 0; k < a[0].size(); k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

/**
 * @brief Function for element-wise multiplication of two vectors.
 * @param a Input vector A.
 * @param b Input vector B.
 * @return Resulting vector after element-wise multiplication.
 * @throws std::invalid_argument if dimensions are incompatible.
 */
std::vector<float> multiply(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) {
        throw std::invalid_argument("Incompatible dimensions for element-wise multiplication. Vectors must have the same size.");
    }
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

/**
 * @brief Function for cross multiplication of a matrix by a vector.
 * Each row of matrix 'b' is multiplied element-wise by vector 'a'.
 * @param a Input vector.
 * @param b Input matrix.
 * @return Resulting matrix after element-wise multiplication.
 * @throws std::invalid_argument if dimensions are incompatible.
 */
std::vector<float> multiply(const std::vector<float>& a, const std::vector<std::vector<float>>& b)
{
    if (b.empty() || a.size() != b.size()) {
        throw std::invalid_argument("Incompatible dimensions for vector-matrix multiplication. Vector size must match matrix row count.");
    }

    // vector to store result
    std::vector<float> result(b[0].size(), 0.0f);
    for(size_t j = 0; j < b[0].size(); j++) {
        // result(j) = a x jth column of b
        for(size_t i = 0; i < b.size(); i++) {
            result[j] += a[i] * b[i][j];
        }
    }
    return result;
}

/**
 * @brief Function for matrix product using threads.
 * @param a Input matrix A.
 * @param b Input matrix B.
 * @return Resulting matrix after element-wise multiplication.
 * @throws std::invalid_argument if dimensions are incompatible.
 */
std::vector<std::vector<float>> multiply(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b)
{
    if (a.size() != b.size() || a.empty() || (!a.empty() && a[0].size() != b[0].size())) {
        throw std::invalid_argument("Incompatible dimensions for element-wise multiplication. Matrices must have the same dimensions.");
    }

    // matrix to store result
    std::vector<std::vector<float>> result(a.size(), std::vector<float>(b[0].size(), 0.0f));
    for(size_t i = 0; i < a.size(); i++) {
        for(size_t j = 0; j < b[0].size(); j++) {
            for(size_t k = 0; k < a[0].size(); k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

/**
 * @brief Function to raise each element of a vector to a specified power.
 * @param input Input vector.
 * @param powerOfValues Power to which each element is raised.
 * @return Resulting vector after exponentiation.
 */
std::vector<float> power(const std::vector<float>& input, const float& powerOfValues)
{
    std::vector<float> result(input.size());
    // each element of input is raised to powerOfValues
    std::transform(input.begin(), input.end(), result.begin(),
                   [powerOfValues](float val) {
                       return std::pow(val, powerOfValues);
                   });
    return result;
}

/**
 * @brief Function to raise each element of a matrix to a specified power.
 * @param input Input matrix.
 * @param powerOfValues Power to which each element is raised.
 * @return Resulting matrix after exponentiation.
 */
std::vector<std::vector<float>> power(const std::vector<std::vector<float>>& input, const float& powerOfValues)
{
    std::vector<std::vector<float>> result(input.size());
    // each element of input is raised to powerOfValues
    std::transform(input.begin(), input.end(), result.begin(),
                   [powerOfValues](const std::vector<float>& row) {
                       std::vector<float> new_row(row.size());
                       std::transform(row.begin(), row.end(), new_row.begin(),
                                      [powerOfValues](float val) { return std::pow(val, powerOfValues); });
                       return new_row;
                   });
    return result;
}

/**
 * @brief Function to perform mean pooling on a matrix.
 * @param input Input matrix.
 * @return Resulting vector after mean pooling.
 */
std::vector<float> meanPool(const std::vector<std::vector<float>>& input) {
    std::vector<float> result(input[0].size(), 0.0f);
    for (const auto& row : input) {
        for (size_t i = 0; i < row.size(); ++i) {
            result[i] += row[i];
        }
    }
    for (auto& val : result) {
        val /= input.size();
    }
    return result;
}

/**
 * @brief Function to perform max pooling on a matrix.
 * @param input Input matrix.
 * @return Resulting vector after max pooling.
 */
std::vector<float> maxPool(const std::vector<std::vector<float>>& input) {
    std::vector<float> result(input[0].size(), std::numeric_limits<float>::lowest());
    for (const auto& row : input) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (row[i] > result[i]) {
                result[i] = row[i];
            }
        }
    }
    return result;
}

/**
 * @brief Function to perform weighted mean pooling on a matrix.
 * @param input Input matrix.
 * @param weights Weights for each row.
 * @return Resulting vector after weighted mean pooling.
 */
std::vector<float> weightedMeanPool(const std::vector<float>& weights, const std::vector<std::vector<float>>& input) {
    if (weights.size() != input[0].size()) {
        throw std::invalid_argument("Weights size must match the number of rows in the input matrix.");
    }

    std::vector<float> result(input.size(), 0.0f);
    result = multiply(weights, input);
    for(int i = 0; i < result.size(); i++) {
        result[i] /= weights.size();
    }
    return result;
}

/**
 * @brief Function to flatten a matrix into a vector.
 * @param input Input matrix.
 * @return Flattened vector.
 */
std::vector<float> flatten(const std::vector<std::vector<float>>& input) {
    std::vector<float> result;
    for (const auto& row : input) {
        result.insert(result.end(), row.begin(), row.end());
    }
    return result;
}

/**
 * @brief Function to reshape a flat vector into a matrix with specified rows and columns.
 * @param input Input flat vector.
 * @param rows Number of rows for the output matrix.
 * @param cols Number of columns for the output matrix.
 * @return Reshaped matrix.
 * @throws std::invalid_argument if the size of input does not match rows * cols.
 */
std::vector<std::vector<float>> reshape(const std::vector<float>& input, int rows, int cols) {
    if (input.size() != rows * cols) {
        throw std::invalid_argument("Input size does not match the specified dimensions for reshape.");
    }

    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = input[i * cols + j];
        }
    }

    return result;
}

/**
 * @brief Function to transpose a matrix.
 * @param input Input matrix.
 * @return Transposed matrix.
 */
std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& input)
{
    std::vector<std::vector<float>> result(input[0].size(), std::vector<float>(input.size()));
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < input[0].size(); ++j) {
            result[j][i] = input[i][j];
        }
    }
    return result;
}

/**
 * @brief average of matrix vector
 * @param input matrix vector
 * @return average matrix
 */
std::vector<std::vector<float>> average(const std::vector<std::vector<std::vector<float>>>& input)
{
    int n = input.size();
    std::vector<std::vector<float>> result(input[0].size(), std::vector<float>(input[0][0].size()));
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < input[i].size(); j++) {
            for(int k = 0; k < input[i][j].size(); k++) {
                result[j][k] += input[i][j][k];
            }
        }
    }
    for(int j = 0; j < result.size(); j++) {
        for(int k = 0; k < result[j].size(); k++) {
            result[j][k] /= n;
        }
    }
    return result;
}

// return maximum element's index in the input vector
int maxIndex(const std::vector<float> &input)
{
    return std::distance(input.begin(), std::max_element(input.begin(), input.end()));
}