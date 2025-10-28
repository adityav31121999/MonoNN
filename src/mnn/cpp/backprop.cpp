#ifdef USE_CPU
#include "mnn.hpp"
#include <numeric>
#include <vector>

/**
 * @brief Backpropagation for the mnn class (1D data).
 * @param target The target output vector.
 */
void mnn::backprop(std::vector<float>& target) {
    // 1. Calculate initial error gradient from the loss function.
    // For sigmoid outputs, Binary Cross-Entropy is the appropriate loss.
    // The derivative of BCE with respect to the output is used here.
    std::vector<float> error_gradient = crossEntropyDer(output, target);

    for (int i = layers - 1; i >= 0; --i) {
        // 2. Multiply error by the derivative of the activation function
        std::vector<float> delta = multiply(error_gradient, sigmoidDer(dotProds[i]));

        // 3. Calculate gradients for c and b weights for the current layer
        const std::vector<float>& prev_activation = (i == 0) ? this->input : activate[i - 1];
        std::vector<float> powered_prev_activation = power(prev_activation, order);

        for (size_t j = 0; j < cweights[i].size(); ++j) {
            for (size_t k = 0; k < cweights[i][j].size(); ++k) {
                cgradients[i][j][k] = powered_prev_activation[j] * delta[k];
                bgradients[i][j][k] = delta[k]; // Gradient for bias is delta, as it's added in the sum
            }
        }

        // 4. Propagate the error to the previous layer
        if (i > 0) {
            // Transpose cweights for backpropagation
            std::vector<std::vector<float>> transposed_cweights(cweights[i][0].size(), std::vector<float>(cweights[i].size()));
            for(size_t r=0; r<cweights[i].size(); ++r) {
                for(size_t c=0; c<cweights[i][0].size(); ++c) {
                    transposed_cweights[c][r] = cweights[i][r][c];
                }
            }
            // Calculate error for the next (previous) layer
            std::vector<float> next_error = delta * transposed_cweights;

            // Multiply by the derivative of the monomial input: d(c*x^n)/dx = n*c*x^(n-1)
            float power_val = order - 1.0f;
            std::vector<float> prev_act_powered_n_minus_1 = power(activate[i-1], power_val);

            // Element-wise multiplication to get the final error gradient for the previous layer's activation
            error_gradient = multiply(next_error, prev_act_powered_n_minus_1);
        }
    }
}

/**
 * @brief Backpropagation for the mnn2d class (2D data).
 * @param target The target output vector (after pooling).
 */
void mnn2d::backprop(std::vector<float> target) {
    // 1. Calculate initial error gradient.
    // For a Softmax output layer combined with Cross-Entropy loss, the gradient
    // with respect to the pre-activation values (before softmax) simplifies to (output - target).
    // Here, 'output' is the result after mean pooling the softmax activations.
    std::vector<float> error_gradient_flat(output.size());
    error_gradient_flat = crossEntropyDer(output, target);

    // Un-pool the gradient. For mean pooling, the gradient is distributed equally.
    const auto& last_activation = activate.back();
    float num_rows = last_activation.size();
    std::vector<std::vector<float>> error_gradient(num_rows, std::vector<float>(last_activation[0].size()));
    for(size_t i = 0; i < num_rows; ++i) {
        for(size_t j = 0; j < last_activation[0].size(); ++j) {
            error_gradient[i][j] = error_gradient_flat[j] / num_rows;
        }
    }

    // The initial 'delta' for the last layer is the un-pooled error gradient itself,
    // because we've already accounted for the softmax and cross-entropy derivative.
    std::vector<std::vector<float>> delta = error_gradient;

    for (int i = layers - 1; i >= 0; --i) {
        // For layers before the last, we calculate delta normally.
        // The delta for the last layer was calculated before the loop.

        // 3. Calculate gradients for c and b weights for the current layer
        const auto& prev_activation = (i == 0) ? this->input : activate[i - 1];
        std::vector<std::vector<float>> powered_prev_activation = power(prev_activation, order);

        // Gradient for cweights: (prev_activation^T) * delta
        for (size_t j = 0; j < cgradients[i].size(); ++j) { // prev_activation cols
            for (size_t k = 0; k < cgradients[i][j].size(); ++k) { // delta cols
                float grad = 0.0f;
                for (size_t r = 0; r < powered_prev_activation.size(); ++r) { // prev_activation rows
                    grad += powered_prev_activation[r][j] * delta[r][k];
                }
                cgradients[i][j][k] = 0.9f * grad;
                bgradients[i][j][k] = 0.1f * grad;
            }
        }

        // 4. Propagate the error to the previous layer
        if (i > 0) {
            // Transpose cweights
            std::vector<std::vector<float>> transposed_cweights(cweights[i][0].size(), std::vector<float>(cweights[i].size()));
            for(size_t r=0; r<cweights[i].size(); ++r) {
                for(size_t c=0; c<cweights[i][0].size(); ++c) {
                    transposed_cweights[c][r] = cweights[i][r][c];
                }
            }

            // Propagate error: error_gradient = (delta * cweights^T)
            std::vector<std::vector<float>> next_error = delta * transposed_cweights;
            
            // Multiply by derivative of the monomial input part: .* (n * prev_activation^(n-1))
            float power_val = order - 1.0f;
            std::vector<std::vector<float>> monomial_der = power(activate[i-1], power_val);
            
            // Calculate the delta for the next (previous) layer for the next iteration
            delta = multiply(next_error, monomial_der);
        }
    }
}

#endif