#ifdef USE_OPENCL
#include "mnn.hpp"
#include <vector>
#include <iostream>
#include <stdexcept>

/**
 * @brief Backpropagation for mnn using OpenCL.
 * @param expected The expected output vector.
 */
void mnn::clBackprop(const std::vector<float>& expected) {
    try {
        cl_int err;
        cl::NDRange local_1d(WORKSIZE_1D);
        size_t inputSize = input.size();
        size_t outputSize = output.size();

        cl::Buffer d_in, d_exp, d_out, d_err;
        std::vector<cl::Buffer> d_incoming(this->layers);
        std::vector<cl::Buffer> d_outgoing(this->layers-1);
        std::vector<cl::Buffer> d_clayers(this->layers);
        std::vector<cl::Buffer> d_blayers(this->layers);
        std::vector<cl::Buffer> d_gradC(this->layers);
        std::vector<cl::Buffer> d_gradB(this->layers);
        std::vector<cl::Buffer> d_dotProds(this->layers);
        std::vector<cl::Buffer> d_activate(this->layers);

        cl::Kernel kernelSigmoidDer, kernelSub, kernelDPow, kernelvxv2m, kernelvxv2v, kernelUpdateWeightsElasticNet;
        kernelSigmoidDer = kernels.at("sigmoidDer");
        kernelSub = kernels.at("subtract");
        kernelDPow = kernels.at("dPower");
        kernelvxv2m = kernels.at("vecxvec2mat");
        kernelvxv2v = kernels.at("vecxvec2vec");
        kernelUpdateWeightsElasticNet = kernels.at("kernelUpdateWeightsElasticNet");

        // load values to buffers
        d_in = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputSize, input.data(), &err); CL_CHECK(err);
        d_exp = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * expected.size(), (void*)expected.data(), &err); CL_CHECK(err);
        d_out = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * output.size(), (void*)output.data(), &err); CL_CHECK(err);
        d_err = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * output.size()); CL_CHECK(err);

        for(int i = 0; i < this->layers; i++) {
            // allot weights to c and b
            size_t cweight_size = cweights[i].size() * cweights[i][0].size();
            size_t bweight_size = bweights[i].size() * bweights[i][0].size();
            d_clayers[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * cweight_size, (void*)flatten(cweights[i]).data(), &err); CL_CHECK(err);
            d_blayers[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * bweight_size, (void*)flatten(bweights[i]).data(), &err); CL_CHECK(err);
            d_gradC[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * cweight_size); CL_CHECK(err);
            d_gradB[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * bweight_size); CL_CHECK(err);
            // fill d_act and d_dotProds
            d_activate[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * activate[i].size(), (void*)activate[i].data(), &err); CL_CHECK(err);
            d_dotProds[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * dotProds[i].size(), (void*)dotProds[i].data(), &err); CL_CHECK(err);
            // allot space to gradients
            d_gradC[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * cweights[i].size() * cweights[i][0].size()); CL_CHECK(err);
            d_gradB[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * bweights[i].size() * bweights[i][0].size()); CL_CHECK(err);
            // for incoming gradients
            d_incoming[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * activate[i].size()); CL_CHECK(err);
        }
        // for outgoing gradients
        for(int i = 0; i < this->layers-1; i++) {
            d_outgoing[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * activate[i].size()); CL_CHECK(err);
        }

        // Calculate initial error (output - expected)
        kernelSub.setArg(0, d_out);
        kernelSub.setArg(1, d_exp);
        kernelSub.setArg(2, d_err);
        kernelSub.setArg(3, (int)outputSize);
        cl::NDRange globalSub = calculate_global_1d(WORKSIZE_1D, outputSize);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSub, cl::NullRange, globalSub, local_1d));

        CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_err, d_incoming[layers - 1], 0, 0, sizeof(float) * outputSize));

        // Backpropagation loop (last layer to second layer)
        for(int layer = layers - 1; layer >= 1; layer--) {
            // outgoing gradient (dL/d(a_{l-1})) using kernelvxv2v
            kernelvxv2v.setArg(0, d_incoming[layer]);                   // incoming gradient (dL/da_l)
            kernelvxv2v.setArg(1, d_clayers[layer]);                    // C weights (C_l)
            kernelvxv2v.setArg(2, d_outgoing[layer - 1]);               // outgoing gradient (dL/da_{l-1})
            kernelvxv2v.setArg(3, (int)activate[layer - 1].size());     // size of the previous layer
            cl::NDRange globalOutGrad = calculate_global_1d(WORKSIZE_1D, activate[layer - 1].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2v, cl::NullRange, globalOutGrad, local_1d));

            // dL/dC_l and dL/dB_l using kernelvxv2m
            kernelvxv2m.setArg(0, d_incoming[layer]);                   // incoming gradient (dL/da_l)
            kernelvxv2m.setArg(1, d_activate[layer - 1]);               // activations of the previous layer (a_{l-1})
            kernelvxv2m.setArg(2, d_gradC[layer]);                      // gradient for C_l (dL/dC_l)
            kernelvxv2m.setArg(3, (int)activate[layer - 1].size());     // size of the previous layer
            kernelvxv2m.setArg(4, (int)cweights[layer].size());         // number of rows in C_l
            cl::NDRange globalWeightGrad = calculate_global_1d(WORKSIZE_1D, cweights[layer].size() * cweights[layer][0].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2m, cl::NullRange, globalWeightGrad, local_1d));

            kernelvxv2m.setArg(0, d_incoming[layer]);                   // incoming gradient (dL/da_l)
            kernelvxv2m.setArg(1, d_blayers[layer]);                    // biases of the current layer (b_l)
            kernelvxv2m.setArg(2, d_gradB[layer]);                      // gradient for B_l (dL/dB_l)
            kernelvxv2m.setArg(3, (int)activate[layer - 1].size());     // size of the previous layer
            kernelvxv2m.setArg(4, (int)bweights[layer].size());         // number of rows in B_l
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2m, cl::NullRange, globalWeightGrad, local_1d));

            // Update weights using kernelUpdateWeightsElasticNet
            kernelUpdateWeightsElasticNet.setArg(0, d_clayers[layer]);              // C weights (C_l)
            kernelUpdateWeightsElasticNet.setArg(1, d_gradC[layer]);                // gradient for C_l (dL/dC_l)
            kernelUpdateWeightsElasticNet.setArg(2, (int)cweights[layer].size());   // number of rows in C_l
            kernelUpdateWeightsElasticNet.setArg(3, (int)activate[layer - 1].size());   // size of the previous layer
            kernelUpdateWeightsElasticNet.setArg(4, learningRate);                  // learning rate
            kernelUpdateWeightsElasticNet.setArg(5, LAMBDA_L1);                     // L1 lambda
            kernelUpdateWeightsElasticNet.setArg(6, LAMBDA_L2);                     // L2 lambda
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdateWeightsElasticNet, cl::NullRange, globalWeightGrad, local_1d));

            kernelUpdateWeightsElasticNet.setArg(0, d_blayers[layer]);              // B weights (B_l)
            kernelUpdateWeightsElasticNet.setArg(1, d_gradB[layer]);                // gradient for B_l (dL/dB_l)
            kernelUpdateWeightsElasticNet.setArg(2, (int)bweights[layer].size());   // number of rows in B_l
            kernelUpdateWeightsElasticNet.setArg(3, (int)activate[layer - 1].size());   // size of the previous layer
            kernelUpdateWeightsElasticNet.setArg(4, learningRate);                  // learning rate
            kernelUpdateWeightsElasticNet.setArg(5, LAMBDA_L1);                     // L1 lambda
            kernelUpdateWeightsElasticNet.setArg(6, LAMBDA_L2);                     // L2 lambda
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdateWeightsElasticNet, cl::NullRange, globalWeightGrad, local_1d));

            // 4. Set the outgoing gradient as the incoming gradient for the next layer
            CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_outgoing[layer - 1], d_incoming[layer - 1], 0, 0, sizeof(float) * activate[layer - 1].size()));
        }

        // Backpropagation for the first layer (input layer)
        // dL/dC_1
        kernelvxv2m.setArg(0, d_incoming[0]);                       // incoming gradient (dL/da_1)
        kernelvxv2m.setArg(1, d_in);                                // input
        kernelvxv2m.setArg(2, d_gradC[0]);                          // gradient for C_1 (dL/dC_1)
        kernelvxv2m.setArg(3, (int)input.size());                   // size of the input
        kernelvxv2m.setArg(4, (int)cweights[0].size());             // number of rows in C_1
        cl::NDRange globalWeightGradFirst = calculate_global_1d(WORKSIZE_1D, cweights[0].size() * cweights[0][0].size());
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2m, cl::NullRange, globalWeightGradFirst, local_1d));
        // dL/dB_1
        kernelvxv2m.setArg(0, d_incoming[0]);                       // incoming gradient (dL/da_1)
        kernelvxv2m.setArg(1, d_blayers[0]);                        // biases of the first layer (b_1)
        kernelvxv2m.setArg(2, d_gradB[0]);                          // gradient for B_1 (dL/dB_1)
        kernelvxv2m.setArg(3, (int)input.size());                   // size of the input
        kernelvxv2m.setArg(4, (int)bweights[0].size());             // number of rows in B_1
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelvxv2m, cl::NullRange, globalWeightGradFirst, local_1d));

        // Update C weights
        kernelUpdateWeightsElasticNet.setArg(0, d_clayers[0]);              // C weights (C_1)
        kernelUpdateWeightsElasticNet.setArg(1, d_gradC[0]);                // gradient for C_1 (dL/dC_1)
        kernelUpdateWeightsElasticNet.setArg(2, (int)cweights[0].size());   // number of rows in C_1
        kernelUpdateWeightsElasticNet.setArg(3, (int)input.size());         // size of the input
        kernelUpdateWeightsElasticNet.setArg(4, learningRate);              // learning rate
        kernelUpdateWeightsElasticNet.setArg(5, LAMBDA_L1);                 // L1 lambda
        kernelUpdateWeightsElasticNet.setArg(6, LAMBDA_L2);                 // L2 lambda
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdateWeightsElasticNet, cl::NullRange, globalWeightGradFirst, local_1d));
        // Update B weights
        kernelUpdateWeightsElasticNet.setArg(0, d_blayers[0]);              // B weights (B_1)
        kernelUpdateWeightsElasticNet.setArg(1, d_gradB[0]);                // gradient for B_1 (dL/dB_1)
        kernelUpdateWeightsElasticNet.setArg(2, (int)bweights[0].size());   // number of rows in B_1
        kernelUpdateWeightsElasticNet.setArg(3, (int)input.size());         // size of the input
        kernelUpdateWeightsElasticNet.setArg(4, learningRate);              // learning rate
        kernelUpdateWeightsElasticNet.setArg(5, LAMBDA_L1);                 // L1 lambda
        kernelUpdateWeightsElasticNet.setArg(6, LAMBDA_L2);                 // L2 lambda
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdateWeightsElasticNet, cl::NullRange, globalWeightGradFirst, local_1d));

        // Read the updated weights back to host
        for (int i = 0; i < this->layers; ++i) {
            size_t cweight_size = cweights[i].size() * cweights[i][0].size();
            size_t bweight_size = bweights[i].size() * bweights[i][0].size();
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_clayers[i], CL_TRUE, 0, sizeof(float) * cweight_size, (void*)flatten(cweights[i]).data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_blayers[i], CL_TRUE, 0, sizeof(float) * bweight_size, (void*)flatten(bweights[i]).data()));
        }
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn::clBackprop: ") + e.what());
    }
}


/**
 * @brief Backpropagation for mnn2d using OpenCL.
 * @param expected The expected output vector.
 */
void mnn2d::clBackprop(const std::vector<float>& expected) {
    try {
        cl_int err;
        cl::NDRange local_1d(WORKSIZE_1D);
        cl::NDRange local_2d(WORKSIZE_2DX, WORKSIZE_2DY);

        // Kernels
        auto kernelSub = kernels.at("subtract");
        auto kernelSoftmaxDer = kernels.at("softmaxDer");
        auto kernelHadamard2 = kernels.at("hadamard2");
        auto kernelMatMul = kernels.at("matxmat2mat");
        auto kernelScale = kernels.at("scaleByValue");
        auto kernelUpdate = kernels.at("kernelUpdateWeightsElasticNet");
        auto kernelTranspose = kernels.at("transpose");
        auto kernelDPow = kernels.at("dPower");
        auto kernelPower = kernels.at("power");
        // --- Buffer Allocation and Initialization ---
        cl::Buffer d_in, d_out, d_exp, d_err;
        std::vector<cl::Buffer> d_clayers(layers), d_blayers(layers), d_gradC(layers), d_gradB(layers);
        std::vector<cl::Buffer> d_dotProds(layers), d_activate(layers), d_incoming(layers);
        std::vector<cl::Buffer> d_outgoing(layers - 1);

        d_in = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size() * input[0].size(), flatten(input).data(), &err); CL_CHECK(err);
        d_out = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * output.size(), output.data(), &err); CL_CHECK(err);
        d_exp = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * expected.size(), (void*)expected.data(), &err); CL_CHECK(err);
        d_err = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * expected.size()); CL_CHECK(err);

        for (int i = 0; i < this->layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            size_t dot_size = dotProds[i].size() * dotProds[i][0].size();
            size_t act_size = activate[i].size() * activate[i][0].size();

            d_clayers[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, c_size * sizeof(float), (void*)flatten(cweights[i]).data(), &err); CL_CHECK(err);
            d_blayers[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, b_size * sizeof(float), (void*)flatten(bweights[i]).data(), &err); CL_CHECK(err);
            d_dotProds[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dot_size * sizeof(float), (void*)flatten(dotProds[i]).data(), &err); CL_CHECK(err);
            d_activate[i] = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, act_size * sizeof(float), (void*)flatten(activate[i]).data(), &err); CL_CHECK(err);
            d_gradC[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, c_size * sizeof(float)); CL_CHECK(err);
            d_gradB[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, b_size * sizeof(float)); CL_CHECK(err);
            d_incoming[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
        }
        for (int i = 0; i < this->layers - 1; ++i) {
            size_t act_size = activate[i].size() * activate[i][0].size();
            d_outgoing[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
        }

        // --- Backpropagation Steps ---

        // Initial error (output - expected)
        kernelSub.setArg(0, d_out);
        kernelSub.setArg(1, d_exp);
        kernelSub.setArg(2, d_err);
        kernelSub.setArg(3, (int)output.size());
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSub, cl::NullRange, calculate_global_1d(WORKSIZE_1D, output.size()), local_1d));

        // error back through mean pooling layer
        std::vector<float> last_layer_error(activate[layers-1].size() * activate[layers-1][0].size());
        std::vector<float> out_err_host(output.size());
        CL_CHECK(clCommandQueue.enqueueReadBuffer(d_err, CL_TRUE, 0, sizeof(float) * output.size(), out_err_host.data()));
        for(size_t i = 0; i < activate[layers-1].size(); ++i) {
            for(size_t j = 0; j < activate[layers-1][0].size(); ++j) {
                last_layer_error[i * activate[layers-1][0].size() + j] = out_err_host[j];
            }
        }
        CL_CHECK(clCommandQueue.enqueueWriteBuffer(d_incoming[layers-1], CL_TRUE, 0, last_layer_error.size() * sizeof(float), last_layer_error.data()));

        // Backpropagate from last to second layer
        for (int layer = layers - 1; layer >= 1; --layer) {
            int prev_rows = activate[layer-1].size();
            int prev_cols = activate[layer-1][0].size();
            int curr_rows = activate[layer].size();
            int curr_cols = activate[layer][0].size();

            // transpose
            cl::Buffer d_C_T(clContext, CL_MEM_READ_WRITE, curr_cols * prev_cols * sizeof(float));
            kernelTranspose.setArg(0, d_clayers[layer]);
            kernelTranspose.setArg(1, d_C_T);
            kernelTranspose.setArg(2, prev_cols);
            kernelTranspose.setArg(3, curr_cols);
            cl::NDRange globalTranspose = calculate_global_2d(new size_t[2]{WORKSIZE_2DX, WORKSIZE_2DY}, prev_cols, curr_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, globalTranspose, local_2d));

            // grad x C^T
            cl::Buffer d_grad_x_CT(clContext, CL_MEM_READ_WRITE, prev_rows * prev_cols * sizeof(float));
            kernelMatMul.setArg(0, d_incoming[layer]);
            kernelMatMul.setArg(1, d_C_T);
            kernelMatMul.setArg(2, d_grad_x_CT);
            kernelMatMul.setArg(3, curr_rows);
            kernelMatMul.setArg(4, curr_cols);
            kernelMatMul.setArg(5, prev_cols);
            cl::NDRange globalMatMul = calculate_global_2d(new size_t[2]{WORKSIZE_2DX, WORKSIZE_2DY}, curr_rows, prev_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, globalMatMul, local_2d));

            //  d(prev_p)
            cl::Buffer d_dprev_p(clContext, CL_MEM_READ_WRITE, prev_rows * prev_cols * sizeof(float));
            kernelDPow.setArg(0, d_activate[layer-1]);
            kernelDPow.setArg(1, d_dprev_p);
            kernelDPow.setArg(2, order);
            kernelDPow.setArg(3, (int)(prev_rows * prev_cols));
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelDPow, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_rows * prev_cols), local_1d));

            // Calculate d(prev_act)
            cl::Buffer d_dprev_act(clContext, CL_MEM_READ_WRITE, prev_rows * prev_cols * sizeof(float));
            size_t prev_dot_size = dotProds[layer-1].size() * dotProds[layer-1][0].size();
            if (prev_dot_size > WORKSIZE_1D) {
                throw std::runtime_error("Softmax derivative kernel cannot process size > WORKSIZE_1D.");
            }
            kernelSoftmaxDer.setArg(0, d_dotProds[layer-1]);
            kernelSoftmaxDer.setArg(1, d_dprev_act);
            kernelSoftmaxDer.setArg(2, 1.1f);
            kernelSoftmaxDer.setArg(3, (int)prev_dot_size);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSoftmaxDer, cl::NullRange, cl::NDRange(prev_dot_size), cl::NDRange(prev_dot_size)));

            // outgoing = (dL/dz_l * C^T) .* d(prev_p) .* d(prev_act)
            kernelHadamard2.setArg(0, d_grad_x_CT);
            kernelHadamard2.setArg(1, d_dprev_p);
            kernelHadamard2.setArg(2, d_dprev_act);
            kernelHadamard2.setArg(3, d_outgoing[layer-1]);
            kernelHadamard2.setArg(4, prev_rows);
            kernelHadamard2.setArg(5, prev_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelHadamard2, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_rows * prev_cols), local_1d));

            // --- Calculate Weight Gradients ---
            // gradc = alpha * prev_p^T * incoming
            // power prev_p
            cl::Buffer d_prev_p(clContext, CL_MEM_READ_WRITE, prev_rows * prev_cols * sizeof(float));
            kernelPower.setArg(0, d_activate[layer-1]);
            kernelPower.setArg(1, d_prev_p);
            kernelPower.setArg(2, order);
            kernelPower.setArg(3, (int)(prev_rows * prev_cols));
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelPower, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_rows * prev_cols), local_1d));

            // transpose prev_p
            cl::Buffer d_prev_p_T(clContext, CL_MEM_READ_WRITE, prev_cols * prev_rows * sizeof(float));
            kernelTranspose.setArg(0, d_prev_p);
            kernelTranspose.setArg(1, d_prev_p_T);
            kernelTranspose.setArg(2, prev_rows);
            kernelTranspose.setArg(3, prev_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, calculate_global_2d(new size_t[2]{WORKSIZE_2DX, WORKSIZE_2DY}, prev_rows, prev_cols), local_2d));

            // dL/dC
            kernelMatMul.setArg(0, d_prev_p_T);
            kernelMatMul.setArg(1, d_incoming[layer]);
            kernelMatMul.setArg(2, d_gradC[layer]);
            kernelMatMul.setArg(3, prev_cols);
            kernelMatMul.setArg(4, prev_rows);
            kernelMatMul.setArg(5, curr_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, calculate_global_2d(new size_t[2]{WORKSIZE_2DX, WORKSIZE_2DY}, prev_cols, curr_cols), local_2d));

            // scale dL/dC
            kernelScale.setArg(0, d_gradC[layer]);
            kernelScale.setArg(1, d_gradC[layer]);
            kernelScale.setArg(2, alpha);
            kernelScale.setArg(3, (int)(prev_cols * curr_cols));
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, calculate_global_1d(WORKSIZE_1D, prev_cols * curr_cols), local_1d));

            // --- Update Weights ---
            cl::NDRange global_update(prev_cols, curr_cols);
            kernelUpdate.setArg(0, d_clayers[layer]);
            kernelUpdate.setArg(1, d_gradC[layer]);
            kernelUpdate.setArg(2, curr_cols);
            kernelUpdate.setArg(3, prev_cols);
            kernelUpdate.setArg(4, learningRate);
            kernelUpdate.setArg(5, LAMBDA_L1);
            kernelUpdate.setArg(6, LAMBDA_L2);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdate, cl::NullRange, global_update, local_2d));

            // Copy outgoing to incoming for the next iteration
            CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_outgoing[layer-1], d_incoming[layer-1], 0, 0, sizeof(float) * prev_rows * prev_cols));
        }

        // --- Backpropagation for the first layer ---
        // (This part is simplified as the CPU version also has a simplified path for the first layer)
        int inHeight = input.size();
        int inWidth = input[0].size();
        int firstLayerRows = activate[0].size();
        int firstLayerCols = activate[0][0].size();

        cl::Buffer d_input(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inHeight * inWidth, (void*)flatten(input).data(), &err); CL_CHECK(err);

        cl::Buffer d_input_p(clContext, CL_MEM_READ_WRITE, inHeight * inWidth * sizeof(float));
        kernelPower.setArg(0, d_input);
        kernelPower.setArg(1, d_input_p);
        kernelPower.setArg(2, order);
        kernelPower.setArg(3, (int)(inHeight * inWidth));
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelPower, cl::NullRange, calculate_global_1d(WORKSIZE_1D, inHeight * inWidth), local_1d));

        cl::Buffer d_input_p_T(clContext, CL_MEM_READ_WRITE, inWidth * inHeight * sizeof(float));
        kernelTranspose.setArg(0, d_input_p);
        kernelTranspose.setArg(1, d_input_p_T);
        kernelTranspose.setArg(2, inHeight);
        kernelTranspose.setArg(3, inWidth);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, calculate_global_2d(new size_t[2]{WORKSIZE_2DX, WORKSIZE_2DY}, inHeight, inWidth), local_2d));

        kernelMatMul.setArg(0, d_input_p_T);
        kernelMatMul.setArg(1, d_incoming[0]);
        kernelMatMul.setArg(2, d_gradC[0]);
        kernelMatMul.setArg(3, inWidth);
        kernelMatMul.setArg(4, inHeight);
        kernelMatMul.setArg(5, firstLayerCols);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelMatMul, cl::NullRange, calculate_global_2d(new size_t[2]{WORKSIZE_2DX, WORKSIZE_2DY}, inWidth, firstLayerCols), local_2d));

        kernelScale.setArg(0, d_gradC[0]);
        kernelScale.setArg(1, d_gradC[0]);
        kernelScale.setArg(2, alpha);
        kernelScale.setArg(3, (int)(inWidth * firstLayerCols));
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, calculate_global_1d(WORKSIZE_1D, inWidth * firstLayerCols), local_1d));

        cl::NDRange global_update_first(inWidth, firstLayerCols);
        kernelUpdate.setArg(0, d_clayers[0]);
        kernelUpdate.setArg(1, d_gradC[0]);
        kernelUpdate.setArg(2, firstLayerCols);
        kernelUpdate.setArg(3, inWidth);
        kernelUpdate.setArg(4, learningRate);
        kernelUpdate.setArg(5, LAMBDA_L1);
        kernelUpdate.setArg(6, LAMBDA_L2);
        CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdate, cl::NullRange, global_update_first, local_2d));

        // --- Read final weights back to host ---
        for (int i = 0; i < layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_clayers[i], CL_TRUE, 0, c_size * sizeof(float), (void*)flatten(cweights[i]).data()));
            // Note: bweights are not updated in this logic, matching the CPU implementation's focus on cweights.
        }

    } catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn2d::clBackprop: ") + e.what());
    }
}


void mnn::clBackprop(const std::vector<std::vector<float>>& expected) {
    try {
        cl_int err;
        int batchSize = expected.size();
        if (batchSize == 0) return;

        // Kernels
        auto kernelAvg = kernels.at("matrix_vector_average");
        auto kernelUpdate = kernels.at("kernelUpdateWeightsElasticNet");
        cl::NDRange local_2d(WORKSIZE_2DX, WORKSIZE_2DY);

        // Buffers
        std::vector<cl::Buffer> d_cweights(layers), d_bweights(layers);
        std::vector<cl::Buffer> d_avg_gradC(layers), d_avg_gradB(layers);

        // Create buffers for weights and average gradients
        for (int i = 0; i < layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            d_cweights[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, c_size * sizeof(float), (void*)flatten(cweights[i]).data(), &err); CL_CHECK(err);
            d_bweights[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, b_size * sizeof(float), (void*)flatten(bweights[i]).data(), &err); CL_CHECK(err);
            d_avg_gradC[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, c_size * sizeof(float)); CL_CHECK(err);
            d_avg_gradB[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, b_size * sizeof(float)); CL_CHECK(err);
        }

        // The CPU implementation sums gradients on the host in the `cgradients` and `bgradients` members.
        // We copy these summed gradients to the device and average them using the `matrix_vector_average` kernel.
        // Note: `matrix_vector_average` expects a 3D buffer (batch of matrices), but we are passing a 2D buffer (summed matrix).
        // By setting the batch size (N) to 1, the kernel will effectively just copy the input, so we must scale it after.
        // A more direct approach is to use a simple scaling kernel.
        for (int layer = 0; layer < layers; ++layer) {
            size_t c_rows = cgradients[layer].size();
            size_t c_cols = cgradients[layer][0].size();
            size_t b_rows = bgradients[layer].size();
            size_t b_cols = bgradients[layer][0].size();

            cl::Buffer d_summed_gradC(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, c_rows * c_cols * sizeof(float), (void*)flatten(cgradients[layer]).data(), &err); CL_CHECK(err);
            cl::Buffer d_summed_gradB(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b_rows * b_cols * sizeof(float), (void*)flatten(bgradients[layer]).data(), &err); CL_CHECK(err);

            // Average the summed gradients: avg_grad = sum_grad / batchSize
            kernelAvg.setArg(0, d_summed_gradC);
            kernelAvg.setArg(1, d_avg_gradC[layer]);
            kernelAvg.setArg(2, batchSize); // The kernel will divide by this N
            kernelAvg.setArg(3, (int)c_rows);
            kernelAvg.setArg(4, (int)c_cols);
            cl::NDRange avg_global_c = calculate_global_2d(new size_t[2]{WORKSIZE_2DX, WORKSIZE_2DY}, c_cols, c_rows);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, avg_global_c, local_2d));

            kernelAvg.setArg(0, d_summed_gradB);
            kernelAvg.setArg(1, d_avg_gradB[layer]);
            kernelAvg.setArg(2, batchSize);
            kernelAvg.setArg(3, (int)b_rows);
            kernelAvg.setArg(4, (int)b_cols);
            cl::NDRange avg_global_b = calculate_global_2d(new size_t[2]{WORKSIZE_2DX, WORKSIZE_2DY}, b_cols, b_rows);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, avg_global_b, local_2d));
        }

        // Update weights for all layers using averaged gradients
        for (int layer = layers - 1; layer >= 0; --layer) {
            int prev_layer_size = (layer == 0) ? inSize : width[layer - 1];
            int curr_layer_size = width[layer];
            cl::NDRange global_update = calculate_global_2d(new size_t[2]{WORKSIZE_2DX, WORKSIZE_2DY}, curr_layer_size, prev_layer_size);

            kernelUpdate.setArg(0, d_cweights[layer]);
            kernelUpdate.setArg(1, d_avg_gradC[layer]);
            kernelUpdate.setArg(2, prev_layer_size); // Corresponds to current_layer_size in kernel
            kernelUpdate.setArg(3, curr_layer_size); // Corresponds to prev_layer_size in kernel
            kernelUpdate.setArg(4, learningRate);
            kernelUpdate.setArg(5, LAMBDA_L1);
            kernelUpdate.setArg(6, LAMBDA_L2);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdate, cl::NullRange, global_update, local_2d));

            kernelUpdate.setArg(0, d_bweights[layer]);
            kernelUpdate.setArg(1, d_avg_gradB[layer]);
            kernelUpdate.setArg(2, prev_layer_size);
            kernelUpdate.setArg(3, curr_layer_size);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdate, cl::NullRange, global_update, local_2d));
        }

        // Read final weights back to host
        for (int i = 0; i < layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_cweights[i], CL_TRUE, 0, c_size * sizeof(float), (void*)flatten(cweights[i]).data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_bweights[i], CL_TRUE, 0, b_size * sizeof(float), (void*)flatten(bweights[i]).data()));
        }

    } catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn::clBackprop (batch): ") + e.what());
    }
}

void mnn2d::clBackprop(const std::vector<std::vector<float>>& expected) {
    try {
        cl_int err;
        int batchSize = expected.size();
        if (batchSize == 0) return;

        // Kernels
        auto kernelAvg = kernels.at("matrix_vector_average");
        auto kernelUpdate = kernels.at("kernelUpdateWeightsElasticNet");
        cl::NDRange local_2d(WORKSIZE_2DX, WORKSIZE_2DY);

        // Buffers
        std::vector<cl::Buffer> d_clayers(layers), d_blayers(layers);
        std::vector<cl::Buffer> d_avg_gradC(layers), d_avg_gradB(layers);

        // Create buffers for weights and average gradients
        for (int i = 0; i < layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            d_clayers[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, c_size * sizeof(float), (void*)flatten(cweights[i]).data(), &err); CL_CHECK(err);
            d_blayers[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, b_size * sizeof(float), (void*)flatten(bweights[i]).data(), &err); CL_CHECK(err);
            d_avg_gradC[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, c_size * sizeof(float)); CL_CHECK(err);
            d_avg_gradB[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, b_size * sizeof(float)); CL_CHECK(err);

            // The CPU version accumulates gradients in the host-side cgradients/bgradients members.
            // We will copy these summed gradients to the device and average them there.
            cl::Buffer d_summed_gradC(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, c_size * sizeof(float), (void*)flatten(cgradients[i]).data(), &err); CL_CHECK(err);
            cl::Buffer d_summed_gradB(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b_size * sizeof(float), (void*)flatten(bgradients[i]).data(), &err); CL_CHECK(err);

            kernelAvg.setArg(0, d_summed_gradC);
            kernelAvg.setArg(1, d_avg_gradC[i]);
            kernelAvg.setArg(2, batchSize); // N
            kernelAvg.setArg(3, (int)cweights[i].size());
            kernelAvg.setArg(4, (int)cweights[i][0].size());
            cl::NDRange avg_global_c = calculate_global_2d(new size_t[2]{WORKSIZE_2DX, WORKSIZE_2DY}, cweights[i][0].size(), cweights[i].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, avg_global_c, local_2d));

            kernelAvg.setArg(0, d_summed_gradB);
            kernelAvg.setArg(1, d_avg_gradB[i]);
            kernelAvg.setArg(2, batchSize); // N
            kernelAvg.setArg(3, (int)bweights[i].size());
            kernelAvg.setArg(4, (int)bweights[i][0].size());
            cl::NDRange avg_global_b = calculate_global_2d(new size_t[2]{WORKSIZE_2DX, WORKSIZE_2DY}, bweights[i][0].size(), bweights[i].size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, avg_global_b, local_2d));
        }

        // Update weights for all layers using the averaged gradients
        for (int i = 0; i < layers; ++i) {
            int current_layer_size = cweights[i].size();
            int prev_layer_size = cweights[i][0].size();
            cl::NDRange global_update = calculate_global_2d(new size_t[2]{WORKSIZE_2DX, WORKSIZE_2DY}, current_layer_size, prev_layer_size);

            kernelUpdate.setArg(0, d_clayers[i]);
            kernelUpdate.setArg(1, d_avg_gradC[i]);
            kernelUpdate.setArg(2, current_layer_size);
            kernelUpdate.setArg(3, prev_layer_size);
            kernelUpdate.setArg(4, learningRate);
            kernelUpdate.setArg(5, LAMBDA_L1);
            kernelUpdate.setArg(6, LAMBDA_L2);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdate, cl::NullRange, global_update, local_2d));

            kernelUpdate.setArg(0, d_blayers[i]);
            kernelUpdate.setArg(1, d_avg_gradB[i]);
            // Args 2-6 are the same
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdate, cl::NullRange, global_update, local_2d));
        }

        // Read final weights back to host
        for (int i = 0; i < layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_clayers[i], CL_TRUE, 0, c_size * sizeof(float), (void*)flatten(cweights[i]).data()));
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_blayers[i], CL_TRUE, 0, b_size * sizeof(float), (void*)flatten(bweights[i]).data()));
        }
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("Exception in mnn2d::clBackprop (batch): ") + e.what());
    }
}

#endif