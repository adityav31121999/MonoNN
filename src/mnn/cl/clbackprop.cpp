#ifdef USE_OPENCL
#include "mnn.hpp"
#include <vector>
#include <stdexcept>

// NOTE: This is a high-level implementation. See note in clforprop.cpp.

/**
 * @brief Backpropagation for mnn using OpenCL.
 * @param expected The expected output vector.
 */
void mnn::clBackprop(const std::vector<float>& expected) {
    // Assumes clForprop was run and device buffers for activations are available.
    // Example:
    // cl::Kernel kernel_bwd1 = cl::Kernel(this->cl_program, "kernelLayerBackward1");
    // cl::Kernel kernel_bwd2 = cl::Kernel(this->cl_program, "kernelLayerBackward2");
    // cl::Kernel kernel_update = cl::Kernel(this->cl_program, "kernelUpdateWeights");

    // 1. Calculate initial error (dL/dz_L) on host or kernel: (output - expected) .* sigmoid_derivative(last_dot_product)
    //    Copy this error to a device buffer `d_incoming_grad`.

    // 2. Loop from the last layer (l = layers-1) down to the second layer (l=1):
    //    - Get layer dimensions: in = width[l-1], out = width[l].
    //    - Get device buffers: d_prev_act, d_C, d_grad_c, d_grad_b, d_outgoing_grad.
    //    - Set kernel_bwd2 args: d_incoming_grad, d_outgoing_grad, d_prev_act, d_C, d_grad_c, d_grad_b, ...
    //    - Enqueue kernel_bwd2 with global size (in, out).
    //    - Set kernel_update args for C and B weights and their gradients.
    //    - Enqueue kernel_update for C and B.
    //    - The `d_outgoing_grad` becomes the `d_incoming_grad` for the next iteration.

    // 3. Handle the first layer (l=0):
    //    - Get layer dimensions: in = inSize, out = width[0].
    //    - Get device buffers: d_input_activations, d_grad_c, d_grad_b.
    //    - Set kernel_bwd1 args: d_incoming_grad, d_input_activations, d_grad_c, d_grad_b, ...
    //    - Enqueue kernel_bwd1 with global size (in, out).
    //    - Enqueue update kernels for C and B weights of the first layer.

    // 4. `queue.finish();`

    throw std::runtime_error("mnn::clBackprop requires full OpenCL context integration.");
}

void mnn::clBackprop(const std::vector<std::vector<float>>& expected) {
    // Batch backprop is more complex. It involves:
    // - Calculating gradients for each item in the batch.
    // - Averaging the gradients across the batch (could be a reduction kernel).
    // - Performing a single weight update step with the averaged gradients.
    throw std::runtime_error("mnn::clBackprop (batch) is not yet implemented.");
}

/**
 * @brief Backpropagation for mnn2d using OpenCL.
 * @param expected The expected output vector.
 */
void mnn2d::clBackprop(const std::vector<float>& expected) {
    // 1. Distribute output error from mean-pooling (can be done on host or kernel).
    //    This means creating a gradient matrix where each element is `(output[i] - expected[i]) / num_elements`.
    // 2. Loop backwards through layers, using a sequence of kernels that implement the logic
    //    of `kernelLayerBackward2D` (matrix multiplies, element-wise products).
    // 3. Update weights at each layer using an update kernel.
    throw std::runtime_error("mnn2d::clBackprop requires full OpenCL context integration.");
}

void mnn2d::clBackprop(const std::vector<std::vector<float>>& expected) {
    throw std::runtime_error("mnn2d::clBackprop (batch) is not yet implemented.");
}
#endif