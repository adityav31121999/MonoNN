// Helper macro/function for parallel reduction within a work-group.
// This is a common pattern for work-group reductions.
// Note: This pattern assumes local_size is a power of 2.
#define WORK_GROUP_REDUCE(OP, INIT_VAL) \
    for (uint s = local_size / 2; s > 0; s /= 2) { \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (local_id < s) { \
            local_buffer[local_id] = OP(local_buffer[local_id], local_buffer[local_id + s]); \
        } \
    }

#ifndef FLT_MAX
    #define FLT_MAX 3.402823466e+38f
#endif

inline float OP_MAX(float a, float b) { return max(a, b); }
inline float OP_MIN(float a, float b) { return min(a, b); }
inline float OP_SUM(float a, float b) { return a + b; }
inline float OP_SUB(float a, float b) { return a - b; }
inline float OP_MUL(float a, float b) { return a * b; }
inline float OP_DIV(float a, float b) { return a / b; }
inline float OP_POW(float a, float b) { return pow(a, b); }
inline float OP_EXP(float a) { return exp(a); }
inline float OP_SIGN(float a) { if (a == 0.0f)
                                    return 0.0f;
                                else
                                    return signbit(a) ? -1.0f : 1.0f;
                              }

/// ----------------- Activations and their Derivatives ----------------- ///

__kernel void sigmoid(__global float* x, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        out[i] = 1.0f / (1.0f + exp(-x[i]));
    }
}

__kernel void sigmoidDer(__global float* x, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        float sigmoid_x = 1.0f / (1.0f + exp(-x[i]));
        out[i] = sigmoid_x * (1.0f - sigmoid_x);
    }
}

// Softmax Kernel
__kernel void softmax(__global float* x, __global float* out, float temp, int size)
{
    int global_id = get_global_id(0); // For global data access (linear index)
    int local_id = get_local_id(0);   // For local memory access
    uint local_size = get_local_size(0); // Work-group size

    // If the entire array must be processed by ONE work-group, 
    // the max local_size (e.g., 256 or 1024) must be >= size.
    // Error handling for size > max_local_size is typically done by the host.
    
    // Using a predefined size for local memory is necessary for OpenCL. 
    // This size must be at least 'local_size' (max expected WGS, e.g., 256).
    // The OpenCL 1.2 standard defines FLT_MAX for the maximum float.

    // Assuming a max work-group size of 256 for the local buffer
    __local float local_buffer[256]; 

    // --- Step 1: Find max value using parallel reduction ---

    // Load data into local memory. Elements outside 'size' get a safe identity value.
    float my_val = (global_id < size) ? x[global_id] : -FLT_MAX; 
    local_buffer[local_id] = my_val;

    barrier(CLK_LOCAL_MEM_FENCE); // Wait for all threads to load their data

    // Perform parallel max reduction (using the defined macro)
    WORK_GROUP_REDUCE(OP_MAX, -FLT_MAX)

    // Thread 0 now has the max value for the work-group
    float max_val = local_buffer[0];

    barrier(CLK_LOCAL_MEM_FENCE); // Wait for max_val to be visible to all threads

    // --- Step 2: Compute exponentials and sum using parallel reduction ---

    // Compute shifted exponential. Elements outside 'size' get 0.0f (identity for sum).
    float shifted_exp = 0.0f;
    if (global_id < size) {
        shifted_exp = exp((x[global_id] - max_val) / temp); 
    }
    
    local_buffer[local_id] = shifted_exp;

    barrier(CLK_LOCAL_MEM_FENCE); // Wait for all threads to compute exponentials

    // Perform parallel sum reduction
    WORK_GROUP_REDUCE(OP_SUM, 0.0f)

    // Thread 0 now has the sum of exponentials for the work-group
    float sum_val = local_buffer[0];

    barrier(CLK_LOCAL_MEM_FENCE); // Wait for sum_val to be visible

    // --- Step 3: Normalize elements and write output ---
    if (global_id < size) {
        if (sum_val > 0.0f) {
            out[global_id] = shifted_exp / sum_val;
        }
        else {
            // Handle sum == 0 case (should ideally not happen with max-shift unless underflow is severe).
            // Setting to a small positive value (e.g., 1/size) or 0.0f is implementation specific.
            // Setting to 0.0f is numerically safer if the result must be 0-1.
            out[global_id] = 0.0f; 
        }
    }
}


// Softmax Derivative Kernel
// This calculates s_i * (1 - s_i) where s_i is the Softmax output.
__kernel void softmaxDer(__global float* x, __global float* out, float temp, int size)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    uint local_size = get_local_size(0);

    __local float local_buffer[256]; // Must be large enough for max local_size

    // --- Step 1 & 2: Identical to Softmax kernel to get max_val and sum_val ---
    
    float my_val = (global_id < size) ? x[global_id] : -FLT_MAX;
    local_buffer[local_id] = my_val;
    barrier(CLK_LOCAL_MEM_FENCE);
    WORK_GROUP_REDUCE(OP_MAX, -FLT_MAX)
    float max_val = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float shifted_exp = 0.0f;
    if (global_id < size) {
        shifted_exp = exp((x[global_id] - max_val) / temp);
    }
    
    local_buffer[local_id] = shifted_exp;
    barrier(CLK_LOCAL_MEM_FENCE);
    WORK_GROUP_REDUCE(OP_SUM, 0.0f)
    float sum_val = local_buffer[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Step 3: Calculate Softmax and its derivative ---
    if (global_id < size) {
        float s_i = 0.0f;
        if (sum_val > 0.0f) {
            s_i = shifted_exp / sum_val; 
        } 
        
        // The derivative for a single element s_i with respect to its own input x_i is s_i * (1 - s_i).
        // This derivative is WRONG if x is the result of a previous operation and we're propagating the gradient.
        // It's correct if the derivative is with respect to the pre-softmax input (for one-hot targets).
        out[global_id] = s_i * (1.0f - s_i);
    }
}

/// ----------------- Math Functions ----------------- ///

__kernel void add(__global float* x, __global float* y, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        out[i] = x[i] + y[i];
    }
}

__kernel void subtract(__global float* x, __global float* y, __global float* out, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        out[i] = x[i] - y[i];
    }
}

__kernel void scaleByValue(__global float* x, __global float* out, float val, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        out[i] = x[i] * val;
    }
}

__kernel void power(__global float* x, __global float* out, float n, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        out[i] = pow(x[i], n);
    }
}

__kernel void dPower(__global float* x, __global float* out, float n, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        out[i] = n * pow(x[i], n - 1.0f);
    }
}

__kernel void meanPool(__global float* in, __global float* out, int inRows, int inCols, int poolSize) {
    int r = get_global_id(0); // Current row in the output
    int c = get_global_id(1); // Current column in the output

    int outRows = inRows / poolSize;
    int outCols = inCols / poolSize;

    if (r < outRows && c < outCols) {
        float sum = 0.0f;
        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                int in_row = r * poolSize + i;
                int in_col = c * poolSize + j;
                sum += in[in_row * inCols + in_col];
            }
        }
        out[r * outCols + c] = sum / (poolSize * poolSize);
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
        out[r * outCols + c] = max_val;
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
        result[i] = x1[row] * x2[col]; // Outer Product
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
        result[i] = sum;
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
        result[i] = sum;
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
        result[i] = sum;
    }
}


__kernel void hadamard (const __global float* mat1, const __global float* mat2, __global float* result, int mat1Rows, int mat1Cols) {
    int i = get_global_id(0);
    int totalSize = mat1Rows * mat1Cols; // Total number of elements in the matrix

    if (i < totalSize) {
        // Element-wise multiplication (Hadamard product)
        result[i] = mat1[i] * mat2[i];
    }
}

__kernel void hadamard2 (const __global float* mat1, const __global float* mat2, const global float* mat3,__global float* result, int mat1Rows, int mat1Cols) {
    int i = get_global_id(0);
    int totalSize = mat1Rows * mat1Cols; // Total number of elements in the matrix

    if (i < totalSize) {
        // Element-wise multiplication (Hadamard product)
        result[i] = mat1[i] * mat2[i] * mat3[i];
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
    outputBuffer[outputIndex] = sum / (float)N;
}


/// ----------------- Forward Propagation ----------------- ///

__kernel void kernelLayerForward1(__global const float* input, __global float* output,
                                  __global const float* cweights, __global const float* bweights,
                                  int inSize, int outSize)
{
    // Each work-item computes one element of the output vector.
    int j = get_global_id(0); // Index for the output vector (column index for cweights/bweights)

    if (j < outSize) {
        float sum = 0.0f;
        for (int i = 0; i < inSize; ++i) {
            // cweights[i][j] -> cweights[i * outSize + j]
            // bweights[i][j] -> bweights[i * outSize + j]
            sum += (input[i] * cweights[i * outSize + j]) + bweights[i * outSize + j];
        }
        // The C++ layerForward uses `output[j] += ...`, implying accumulation.
        // We assume 'output' is pre-initialized (e.g., to zeros) on the host
        // or that this kernel is meant to accumulate into existing values.
        output[j] += sum;
    }
}

__kernel void kernelLayerForward2(__global const float* input, __global float* output,
                                  __global const float* cweights, __global const float* bweights,
                                  int inSize, int outSize, float n)
{
    // Each work-item computes one element of the output vector.
    int j = get_global_id(0); // Index for the output vector (column index for cweights/bweights)

    if (j < outSize) {
        float sum = 0.0f;
        for (int i = 0; i < inSize; ++i) {
            // Calculate powerIn[i] = pow(input[i], n)
            float powered_input_i = pow(input[i], n);
            // cweights[i][j] -> cweights[i * outSize + j]
            // bweights[i][j] -> bweights[i * outSize + j]
            sum += (powered_input_i * cweights[i * outSize + j]) + bweights[i * outSize + j];
        }
        // The C++ layerForward uses `output[j] += ...`, implying accumulation.
        // We assume 'output' is pre-initialized (e.g., to zeros) on the host
        // or that this kernel is meant to accumulate into existing values.
        output[j] += sum;
    }
}

__kernel void kernelLayerForward3(__global const float* input, __global float* output,
                                  __global const float* cweights, __global const float* bweights,
                                  int inHeight, int inWidth, int outSize)
{
    // Each work-item computes one element of the output matrix.
    // Global ID 0 for row index, Global ID 1 for column index.
    int i = get_global_id(0); // Row index for input, output, bweights
    int j = get_global_id(1); // Column index for output, cweights, bweights

    // The C++ code's error checks and usage for bweights are inconsistent.
    // Following the usage `output[i][j] = ... + bweights[i][j]`,
    // we assume bweights is an `inHeight x outSize` matrix.
    // cweights is `inWidth x outSize`.
    // input is `inHeight x inWidth`.
    // output is `inHeight x outSize`.

    if (i < inHeight && j < outSize) {
        float dotProd_ij = 0.0f;
        for (int k = 0; k < inWidth; ++k) { // k iterates over inWidth (cweights_rows)
            // input[i][k] -> input[i * inWidth + k]
            // cweights[k][j] -> cweights[k * outSize + j]
            dotProd_ij += input[i * inWidth + k] * cweights[k * outSize + j];
        }
        // bweights[i][j] -> bweights[i * outSize + j]
        // The C++ layerForward uses `output[i][j] = ...`, implying assignment.
        output[i * outSize + j] = dotProd_ij + bweights[i * outSize + j];
    }
}

__kernel void kernelLayerForward4(__global const float* input, __global float* output,
                                  __global const float* cweights, __global const float* bweights,
                                  int inHeight, int inWidth, int outSize, float n)
{
    // Each work-item computes one element of the output matrix.
    // Global ID 0 for row index, Global ID 1 for column index.
    int i = get_global_id(0); // Row index for input, output, bweights
    int j = get_global_id(1); // Column index for output, cweights, bweights

    // The C++ code's error checks and usage for bweights are inconsistent.
    // Following the usage `output[i][j] = ... + bweights[i][j]`,
    // we assume bweights is an `inHeight x outSize` matrix.
    // cweights is `inWidth x outSize`.
    // input is `inHeight x inWidth`.
    // output is `inHeight x outSize`.

    if (i < inHeight && j < outSize) {
        float dotProd_ij = 0.0f;
        for (int k = 0; k < inWidth; ++k) { // k iterates over inWidth (cweights_rows)
            // input[i][k] -> input[i * inWidth + k]
            // powerIn[i][k] is pow(input[i * inWidth + k], n)
            // cweights[k][j] -> cweights[k * outSize + j]
            dotProd_ij += pow(input[i * inWidth + k], n) * cweights[k * outSize + j];
        }
        // bweights[i][j] -> bweights[i * outSize + j]
        // The C++ layerForward uses `output[i][j] = ...`, implying assignment.
        output[i * outSize + j] = dotProd_ij + bweights[i * outSize + j];
    }
}

/// ----------------- Backpropagation ----------------- ///



/// ----------------- Update Weights ----------------- ///

__kernel void kernelUpdateWeights(__global float* weights, 
                                __global float* gweights,
                                float learning_rate,
                                int current_layer_size,
                                int prev_layer_size)
{
    // Let global_id(0) be column (previous layer) and global_id(1) be row (current layer)
    int i = get_global_id(0); // Index for previous layer size (column)
    int j = get_global_id(1); // Index for current layer size (row)

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i;
        // The original kernel had `gweights[weight_idx] = gradient;` which was incomplete.
        // Assuming gweights already contains the gradient for this weight.
        weights[weight_idx] -= learning_rate * gweights[weight_idx];
    }
}


__kernel void kernelUpdateWeightsWithL1(__global float* weights,
                                        __global float* gweights, 
                                        int current_layer_size, 
                                        int prev_layer_size,
                                        float learning_rate, 
                                        float lambda_l1)
{
    // Let global_id(0) be column (previous layer) and global_id(1) be row (current layer)
    int i = get_global_id(0); // Index for previous layer size (column)
    int j = get_global_id(1); // Index for current layer size (row)

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i; // Row-major index

        float current_weight = weights[weight_idx];

        // Subgradient of the L1 regularization term
        float sign;
        if (current_weight < 0.0f) {
            sign = -1.0f;
        }
        else if (current_weight > 0.0f) {
            sign = 1.0f;
        }
        else {
            sign = 0.0f;
        }

        float l1_gradient = sign * lambda_l1;

        // Total gradient
        float total_gradient = gweights[weight_idx] + l1_gradient;

        weights[weight_idx] -= learning_rate * total_gradient;
    }
}


__kernel void kernelUpdateWeightsWithL2(__global float* weights,
                                        __global float* gweights, 
                                        int current_layer_size, 
                                        int prev_layer_size,
                                        float learning_rate, 
                                        float lambda_l2)
{
    // Let global_id(0) be column (previous layer) and global_id(1) be row (current layer)
    int i = get_global_id(0); // Index for previous layer size (column)
    int j = get_global_id(1); // Index for current layer size (row)

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i; // Row-major index
        
        // Gradient of the L2 regularization term
        float current_weight = weights[weight_idx];
        float l2_gradient = lambda_l2 * current_weight;
        
        // Total gradient
        float total_gradient = gweights[weight_idx] + l2_gradient;

        weights[weight_idx] -= learning_rate * total_gradient;
    }
}


__kernel void kernelUpdateWeightsElasticNet(__global float* weights,
                                            __global float* gweights, 
                                            int current_layer_size, 
                                            int prev_layer_size,
                                            float learning_rate, 
                                            float lambda_l1,
                                            float lambda_l2)
{
    // Let global_id(0) be column (previous layer) and global_id(1) be row (current layer)
    int i = get_global_id(0); // Index for previous layer size (column)
    int j = get_global_id(1); // Index for current layer size (row)

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i; // Row-major index
        
        // Gradient of the L2 regularization term
        float current_weight = weights[weight_idx];
        float l2_gradient = lambda_l2 * current_weight;
        
        // Subgradient of the L1 regularization term
        float sign;
        if (current_weight < 0.0f) {
            sign = -1.0f;
        }
        else if (current_weight > 0.0f) {
            sign = 1.0f;
        }
        else {
            sign = 0.0f; // Subgradient is 0 when weight is 0
        }

        float l1_gradient = sign * lambda_l1;

        // Total gradient
        float total_gradient = gweights[weight_idx] + l2_gradient + l1_gradient;

        weights[weight_idx] -= learning_rate * total_gradient;
    }
}


__kernel void kernelUpdateWeightsWithWeightDecay(__global float* weights,
                                                 __global float* gweights,
                                                 int current_layer_size,
                                                 int prev_layer_size,
                                                 float learning_rate,
                                                 float decay_rate)
{
    // Let global_id(0) be column (previous layer) and global_id(1) be row (current layer)
    int i = get_global_id(0); // Index for previous layer size (column)
    int j = get_global_id(1); // Index for current layer size (row)

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i; // Row-major index

        float current_weight = weights[weight_idx];
        float gradient = gweights[weight_idx];

        weights[weight_idx] = current_weight * (1.0f - decay_rate) - learning_rate * gradient;
    }
}


__kernel void kernelUpdateWeightsDropout(__global float* weights,
                                         __global float* gweights,
                                         int current_layer_size,
                                         int prev_layer_size,
                                         float learning_rate,
                                         float dropout_rate,
                                         uint base_seed)
{
    // Let global_id(0) be column (previous layer) and global_id(1) be row (current layer)
    int i = get_global_id(0); // Index for previous layer size (column)
    int j = get_global_id(1); // Index for current layer size (row)

    if (i < prev_layer_size && j < current_layer_size) {
        int weight_idx = j * prev_layer_size + i; // Row-major index

        // Create a unique seed for each work-item to ensure different random numbers.
        // The base_seed should be changed for each kernel call (e.g., based on time or iteration).
        uint seed = base_seed + weight_idx;

        // Simple Linear Congruential Generator (LCG) for pseudo-random numbers.
        seed = (seed * 1664525u + 1013904223u);
        float rand_val = (float)(seed) / (float)(0xFFFFFFFF); // Normalize to [0.0, 1.0]

        if (rand_val >= dropout_rate) {
            weights[weight_idx] -= learning_rate * gweights[weight_idx];
        }
        // If rand_val < dropout_rate, the weight is "dropped" and remains unchanged.
    }
}
