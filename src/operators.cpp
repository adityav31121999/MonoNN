#include "include/mnn.hpp"
#include <vector>
#include <stdexcept>
#include <thread>
#include <algorithm>

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
 * @brief Function for cross multiplication of a matrix by a vector.
 * Each row of matrix 'b' is multiplied element-wise by vector 'a'.
 * @param a Input vector.
 * @param b Input matrix.
 * @return Resulting matrix after element-wise multiplication.
 * @throws std::invalid_argument if dimensions are incompatible.
 */
std::vector<float> multiply(const std::vector<float>& a, const std::vector<std::vector<float>>& b)
{
    if (b.empty() || a.size() != b[0].size()) {
        throw std::invalid_argument("Incompatible dimensions for element-wise multiplication. Vector size must match matrix column size.");
    }
 
    std::vector<float> result(b.size(), 0.0f);
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    num_threads = std::min((unsigned int)b.size(), num_threads > 0 ? num_threads : 1);
 
    auto worker = [&](size_t start_row, size_t end_row) {
        for (size_t i = start_row; i < end_row; ++i) {
            for (size_t j = 0; j < a.size(); ++j) {
                result[i] += a[j] * b[i][j];
            }
        }
    };
 
    size_t rows_per_thread = b.size() / num_threads;
    for (unsigned int i = 0; i < num_threads; ++i) {
        size_t start = i * rows_per_thread;
        size_t end = (i == num_threads - 1) ? b.size() : start + rows_per_thread;
        threads.emplace_back(worker, start, end);
    }
 
    for (auto& t : threads) {
        t.join();
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
 
    std::vector<std::vector<float>> result(a.size(), std::vector<float>(a[0].size()));
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    num_threads = std::min((unsigned int)a.size(), num_threads > 0 ? num_threads : 1);
 
    auto worker = [&](size_t start_row, size_t end_row) {
        for (size_t i = start_row; i < end_row; ++i) {
            for (size_t j = 0; j < a[0].size(); ++j) {
                result[i][j] = a[i][j] * b[i][j];
            }
        }
    };
 
    size_t rows_per_thread = a.size() / num_threads;
    for (unsigned int i = 0; i < num_threads; ++i) {
        size_t start = i * rows_per_thread;
        size_t end = (i == num_threads - 1) ? a.size() : start + rows_per_thread;
        threads.emplace_back(worker, start, end);
    }
 
    for (auto& t : threads) {
        t.join();
    }
 
    return result;
}

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
