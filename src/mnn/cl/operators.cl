/// ----------------- Math Functions ----------------- ///

__kernel void add(__global float* x, __global float* y, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        out[i] = valueCorrection(x[i] + y[i]);
    }
}

__kernel void subtract(__global float* x, __global float* y, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        out[i] = valueCorrection(x[i] - y[i]);
    }
}

__kernel void scaleByValue(__global float* x, __global float* out, float val, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        out[i] = valueCorrection(x[i] * val);
    }
}

__kernel void power(__global float* x, __global float* out, float n, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        out[i] = valueCorrection(pow(x[i], n));
    }
}

__kernel void dPower(__global float* x, __global float* out, float n, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        out[i] = valueCorrection(n * pow(x[i], n - 1.0f));
    }
}


__kernel void meanPool(__global float* in, __global float* out, int inRows, int inCols, int poolSize)
{
    int c = get_global_id(0); // c is the column index.
    if (c < inCols) {
        float sum = 0.0f;
        for (int r = 0; r < inRows; ++r) {
            sum += in[r * inCols + c];
        }
        out[c] = valueCorrection(sum / (float)inRows);
    }
}


__kernel void maxPool(__global float* in, __global float* out, int inRows, int inCols, int poolSize) {
    int r = get_global_id(0); // Current row in the output
    int c = get_global_id(1); // Current column in the output

    int outRows = inRows / poolSize;
    int outCols = inCols / poolSize;

    if (r < outRows && c < outCols) {
        float max_val = -FLT_MAX; // Initialize with a very small number
        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                int in_row = r * poolSize + i;
                int in_col = c * poolSize + j;
                max_val = max(max_val, in[in_row * inCols + in_col]);
            }
        }
        out[r * outCols + c] = valueCorrection(max_val);
    }
}

__kernel void transpose (const __global float* in, __global float* out, int rows, int cols) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < rows && j < cols) {
        out[j * rows + i] = in[i * cols + j];
    }
}

__kernel void vecxvec2vec (const __global float* x1, const __global float* x2, __global float* result, int size) {
    int i = get_global_id(0);
    if (i < size) {
        result[i] = x1[i] * x2[i];
    }
}

__kernel void vecxvec2mat (const __global float* x1, const __global float* x2, __global float* result, int x1size, int x2size) {
    int i = get_global_id(0);
    int resultSize = x1size * x2size;
    if (i < resultSize) {
        int row = i / x2size;
        int col = i % x2size;
        result[i] = valueCorrection(x1[row] * x2[col]); // Outer Product
    }
}

__kernel void vecxmat2vec (const __global float* vec, const __global float* mat, __global float* result, int matRows, int matCols) {
    int i = get_global_id(0);
    // i is the column index of the result vector (0 to matCols-1)
    if (i < matCols) {
        float sum = 0.0f; // Initialize accumulator
        // j is the index of vec / row index of mat (0 to matRows-1)
        for (int j = 0; j < matRows; j++) {
            // mat[j][i] index is j * matCols + i
            sum += vec[j] * mat[j * matCols + i];
        }
        result[i] = valueCorrection(sum);
    }
}

__kernel void matxmat2mat (const __global float* mat1, const __global float* mat2,
                            __global float* result, int mat1Rows, int mat1Cols, int mat2cols) 
{
    int i = get_global_id(0);
    int resultSize = mat1Rows * mat2cols; // Total size of the result matrix (M x N)

    if (i < resultSize) {
        int r = i / mat2cols; // Row index in mat1 and result (M)
        int c = i % mat2cols; // Col index in mat2 and result (N)
        int K = mat1Cols;     // Inner dimension (K)

        float sum = 0.0f; // Initialize accumulator
        for (int k = 0; k < K; k++) {
            // result[r][c] = sum(mat1[r][k] * mat2[k][c])
            // mat1[r][k] index: r * mat1Cols + k
            // mat2[k][c] index: k * mat2cols + c
            sum += mat1[r * mat1Cols + k] * mat2[k * mat2cols + c];
        }
        result[i] = valueCorrection(sum);
    }
}

__kernel void matxvec2vec (const __global float* mat, const __global float* vec, __global float* result, int matRows, int matCols) {
    int i = get_global_id(0);
    // i is the row index of mat / element index of result (0 to matRows-1)
    if (i < matRows) {
        float sum = 0.0f; // Initialize accumulator
        // j is the column index of mat / element index of vec (0 to matCols-1)
        for (int j = 0; j < matCols; j++) {
            // mat[i][j] index: i * matCols + j
            sum += mat[i * matCols + j] * vec[j];
        }
        result[i] = valueCorrection(sum);
    }
}


__kernel void hadamard (const __global float* mat1, const __global float* mat2, __global float* result, int mat1Rows, int mat1Cols) {
    int i = get_global_id(0);
    int totalSize = mat1Rows * mat1Cols; // Total number of elements in the matrix

    if (i < totalSize) {
        // Element-wise multiplication (Hadamard product)
        result[i] = valueCorrection(mat1[i] * mat2[i]);
    }
}

__kernel void hadamard2 (const __global float* mat1, const __global float* mat2, const global float* mat3,
        __global float* result, int mat1Rows, int mat1Cols)
{
    int i = get_global_id(0);
    int totalSize = mat1Rows * mat1Cols; // Total number of elements in the matrix

    if (i < totalSize) {
        // Element-wise multiplication (Hadamard product)
        result[i] = valueCorrection(mat1[i] * mat2[i] * mat3[i]);
    }
}


/**
 * @brief Computes the element-wise average of a vector of matrices.
 *
 * Each work-item is responsible for calculating one element (j, k) of the
 * final averaged matrix. It does this by iterating through all 'N' input
 * matrices, summing the values at its assigned (j, k) position, and then

 * dividing by N.
 *
 * @param inputBuffer   A flattened 1D buffer representing the 3D input data
 *                      (N matrices of size Rows x Cols).
 * @param outputBuffer  A flattened 1D buffer to store the 2D averaged result.
 * @param N             The number of matrices in the input vector.
 * @param Rows          The number of rows in each matrix.
 * @param Cols          The number of columns in each matrix.
 */
__kernel void matrix_vector_average(
    __global const float* inputBuffer,
    __global float* outputBuffer,
    const int N,
    const int Rows,
    const int Cols)
{
    // Identify the position (row j, column k) this work-item is responsible for.
    // get_global_id(0) corresponds to the columns (width).
    // get_global_id(1) corresponds to the rows (height).
    int k = get_global_id(0); // Column index
    int j = get_global_id(1); // Row index

    // Boundary check to ensure we don't process out of bounds.
    if (j >= Rows || k >= Cols) {
        return;
    }

    float sum = 0.0f;
    int matrixSize = Rows * Cols;

    // Loop through all 'N' matrices.
    for (int i = 0; i < N; ++i) {
        // Calculate the 1D index for input[i][j][k] in the flattened buffer.
        int inputIndex = i * matrixSize + j * Cols + k;
        sum += inputBuffer[inputIndex];
    }

    // Calculate the 1D index for the output[j][k] in the flattened buffer.
    int outputIndex = j * Cols + k;

    // Calculate the average and write it to the output buffer.
    outputBuffer[outputIndex] = valueCorrection(sum / (float)N);
}

/**
 * @brief Sums all elements of a matrix and stores the result in a single-element output buffer.
 *
 * @param inputBuffer   A flattened 1D buffer representing the input matrix.
 * @param outputBuffer  A single-element buffer to store the sum.
 * @param Rows          The number of rows in the matrix.
 * @param Cols          The number of columns in the matrix.
 */
__kernel void matrix_vector_sum(
    __global const float* inputBuffer,
    __global float* outputBuffer,
    const int Rows,
    const int Cols)
{
    // This is a simple, non-parallelized sum for demonstration.
    // For large matrices, a parallel reduction would be more efficient.
    if (get_global_id(0) == 0 && get_global_id(1) == 0) {
        float sum = 0.0f;
        for (int i = 0; i < Rows * Cols; ++i) {
            sum += inputBuffer[i];
        }
        outputBuffer[0] = valueCorrection(sum);
    }
}

/// ----------------- Update Weights ----------------- ///

__kernel void kernelUpdateWeights(__global float* weights, 
                                __global float* gweights,
                                float learning_rate,
                                int totalElements)
{
    int i = get_global_id(0);
    if (i < totalElements) {
        weights[i] = valueCorrection(weights[i] - learning_rate * gweights[i]);
    }
}

__kernel void kernelUpdateWeightsWithL1(__global float* weights,
                                        __global float* gweights, 
                                        int totalElements,
                                        float learning_rate, 
                                        float lambda_l1)
{
    int i = get_global_id(0);
    if (i < totalElements) {
        float current_weight = weights[i];

        // Subgradient of the L1 regularization term
        float sign = OP_SIGN(current_weight);

        float l1_gradient = sign * lambda_l1;

        // Total gradient
        float total_gradient = gweights[i] + l1_gradient;

        weights[i] = valueCorrection(weights[i] - learning_rate * total_gradient);
    }
}


__kernel void kernelUpdateWeightsWithL2(__global float* weights,
                                        __global float* gweights, 
                                        int totalElements,
                                        float learning_rate, 
                                        float lambda_l2)
{
    int i = get_global_id(0);
    if (i < totalElements) {
        // Gradient of the L2 regularization term
        float current_weight = weights[i];
        float l2_gradient = lambda_l2 * current_weight;
        
        // Total gradient
        float total_gradient = gweights[i] + l2_gradient;

        weights[i] = valueCorrection(weights[i] - learning_rate * total_gradient);
    }
}


__kernel void kernelUpdateWeightsElasticNet(__global float* weights,
                                            __global float* gweights, 
                                            int totalElements,
                                            float learning_rate, 
                                            float lambda_l1,
                                            float lambda_l2)
{
    int i = get_global_id(0);
    if (i < totalElements) {
        // Gradient of the L2 regularization term
        float current_weight = weights[i];
        float l2_gradient = lambda_l2 * current_weight;
        
        // Subgradient of the L1 regularization term
        float sign = OP_SIGN(current_weight);

        float l1_gradient = sign * lambda_l1;

        // Total gradient
        float total_gradient = gweights[i] + l2_gradient + l1_gradient;

        weights[i] = valueCorrection(weights[i] - learning_rate * total_gradient);
    }
}


__kernel void kernelUpdateWeightsWithWeightDecay(__global float* weights,
                                                 __global float* gweights,
                                                 int totalElements,
                                                 float learning_rate,
                                                 float decay_rate)
{
    int i = get_global_id(0);
    if (i < totalElements) {
        float current_weight = weights[i];
        float gradient = gweights[i];

        weights[i] = valueCorrection(current_weight * (1.0f - decay_rate) - learning_rate * gradient);
    }
}


__kernel void kernelUpdateWeightsDropout(__global float* weights,
                                         __global float* gweights,
                                         int totalElements,
                                         float learning_rate,
                                         float dropout_rate,
                                         uint base_seed)
{
    int i = get_global_id(0);
    if (i < totalElements) {
        // Create a unique seed for each work-item to ensure different random numbers.
        // The base_seed should be changed for each kernel call (e.g., based on time or iteration).
        uint seed = base_seed + i;

        // Simple Linear Congruential Generator (LCG) for pseudo-random numbers.
        seed = (seed * 1664525u + 1013904223u);
        float rand_val = (float)(seed) / (float)(0xFFFFFFFF); // Normalize to [0.0, 1.0]

        if (rand_val >= dropout_rate) {
            weights[i] = valueCorrection(weights[i] - learning_rate * gweights[i]);
        }
        // If rand_val < dropout_rate, the weight is "dropped" and remains unchanged.
    }
}
