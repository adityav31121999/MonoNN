#include "operators.hpp"
#include <thread>
#include <mutex>

/**
 * @brief Function for element-wise multiplication or hadamard product.
 * @param a Input matrix A.
 * @param b Input matrix B.
 * @return Resulting matrix after element-wise multiplication.
 * @throws std::invalid_argument if dimensions are incompatible.
 */
std::vector<std::vector<float>> hadamard(const std::vector<std::vector<float>> &a, const std::vector<std::vector<float>> &b)
{
    if(a.size() != b.size() || a[0].size() != b.size()) {
        throw std::runtime_error("Matrix dimensions incompatible for hadamard product: " 
                                 + std::to_string(a.size()) + " x " + std::to_string(a[0].size())
                                 + " v/s "
                                 + std::to_string(b.size()) + " x " + std::to_string(b[0].size()));
    }

    std::vector<std::vector<float>> result(a.size(), std::vector<float>(a[0].size(), 0.0f));
    for(int i = 0; i < a.size(); i++) {
        for(int j = 0; j < a[0].size(); j++) {
            result[i][j] = a[i][j] * b[i][j];
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
 * @brief Function for cross multiplication of vector and matrix.
 * Each row of matrix 'b' is multiplied element-wise by vector 'a'.
 * @param a Input vector.
 * @param b Input matrix.
 * @return Resulting vector after element-wise multiplication.
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
 * @brief Function for matrix multiplication (standard matrix product).
 * @param a Input matrix A.
 * @param b Input matrix B.
 * @return Resulting matrix after matrix multiplication.
 * @throws std::invalid_argument if dimensions are incompatible.
 */
std::vector<std::vector<float>> multiply(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b)
{
    // Standard matrix multiplication
    if (a[0].size() == b.size()) {
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
    else {
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication.");
    }
}

// thread support for faster execution

/**
 * @brief Function for cross multiplication of a vector and matrix using threads.
 * Each row of matrix 'b' is multiplied element-wise by vector 'a'.
 * Strategy: Parallelize the outer loop (columns of result).
 */
std::vector<float> multiplyWithThreads(const std::vector<float>& a, const std::vector<std::vector<float>>& b)
{
    if (b.empty() || a.size() != b.size()) {
        throw std::invalid_argument("Incompatible dimensions for vector-matrix multiplication. Vector size must match matrix row count.");
    }

    size_t num_cols = b[0].size();
    size_t vec_size = a.size();

    // Vector to store result
    std::vector<float> result(num_cols, 0.0f);

    // Determine number of threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    // Fallback if hardware_concurrency returns 0, and don't create more threads than columns
    num_threads = (num_threads == 0) ? 2 : std::min(num_threads, (unsigned int)num_cols);

    std::vector<std::thread> threads;
    size_t chunk_size = num_cols / num_threads;

    // Lambda to process a specific range of columns
    auto worker = [&](size_t start_col, size_t end_col) {
        for(size_t j = start_col; j < end_col; j++) {
            // result(j) = a x jth column of b
            // We use a local sum to minimize cache thrashing on result[j]
            float sum = 0.0f;
            for(size_t i = 0; i < vec_size; i++) {
                sum += a[i] * b[i][j];
            }
            result[j] = sum;
        }
    };

    // Launch threads
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        // The last thread handles the remaining columns if num_cols isn't perfectly divisible
        size_t end = (t == num_threads - 1) ? num_cols : start + chunk_size;
        
        threads.emplace_back(worker, start, end);
    }

    // Wait for all threads to finish
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    return result;
}

/**
 * @brief Function for matrix multiplication using threads.
 * Strategy: Parallelize the rows of the resulting matrix (Result A).
 */
std::vector<std::vector<float>> multiplyWithThreads(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b)
{
    if (a.empty() || b.empty() || a[0].size() != b.size()) {
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication.");
    }

    size_t rows_a = a.size();
    size_t cols_b = b[0].size();
    size_t common_dim = b.size(); // equals a[0].size()

    std::vector<std::vector<float>> result(rows_a, std::vector<float>(cols_b, 0.0f));

    // Determine number of threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    // Don't create more threads than there are rows to process
    num_threads = (num_threads == 0) ? 2 : std::min(num_threads, (unsigned int)rows_a);

    std::vector<std::thread> threads;
    size_t chunk_size = rows_a / num_threads;

    // Lambda to process a specific range of rows of Matrix A
    auto worker = [&](size_t start_row, size_t end_row) {
        for(size_t i = start_row; i < end_row; i++) {
            for(size_t j = 0; j < cols_b; j++) {
                // Perform dot product
                float sum = 0.0f;
                for(size_t k = 0; k < common_dim; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
    };

    // Launch threads
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        // The last thread handles the remainder
        size_t end = (t == num_threads - 1) ? rows_a : start + chunk_size;

        threads.emplace_back(worker, start, end);
    }

    // Wait for completion
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    return result;
}