#ifdef USE_CL
#include "mnn1d.hpp"
#include "mnn2d.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm> // For std::max
#include <cmath>     // For std::ceil
#include <limits>    // For std::numeric_limits

/**
 * @brief Trains the mnn network on a single input-target pair with optimized buffer management.
 * @param input The input vector.
 * @param target The target output vector.
 */
void mnn1d::clBufTrain(const std::vector<float>& input, const std::vector<float>& target) {
    cl_int err;
    cl::NDRange local_1d(WORKSIZE_1D);
    cl::NDRange local_2d(WORKSIZE_2DX, WORKSIZE_2DY);
    size_t size2d[2] = {WORKSIZE_2DX, WORKSIZE_2DY};

    // --- Buffer Allocation ---
    cl::Buffer d_in, d_exp, d_out, d_err;
    std::vector<cl::Buffer> d_cweights(layers), d_bweights(layers), d_gradC(layers), d_gradB(layers);
    std::vector<cl::Buffer> d_dotProds(layers), d_activate(layers), d_incoming(layers);
    cl::Buffer d_ones, d_preoutgoing_l, d_outgoing_l, d_dpow_l, d_dact_l;
    cl::Buffer d_C_T_buf; // Reusable buffer for transposed weights

    try {
        // Allocate input/output/target buffers
        d_in = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.size() * sizeof(float), (void*)input.data(), &err); CL_CHECK(err);
        d_exp = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, target.size() * sizeof(float), (void*)target.data(), &err); CL_CHECK(err);
        d_out = cl::Buffer(clContext, CL_MEM_READ_WRITE, output.size() * sizeof(float)); CL_CHECK(err);
        d_err = cl::Buffer(clContext, CL_MEM_READ_WRITE, output.size() * sizeof(float)); CL_CHECK(err);

        size_t max_layer_width = 0;
        for (int w : width) max_layer_width = std::max(max_layer_width, (size_t)w);
        max_layer_width = std::max(max_layer_width, input.size());
        max_layer_width = std::max(max_layer_width, output.size());
        std::vector<float> v1(max_layer_width, 1.0f);
        d_ones = cl::Buffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * max_layer_width, v1.data(), &err); CL_CHECK(err);

        size_t max_weight_size = 0;
        for(const auto& w : cweights) max_weight_size = std::max(max_weight_size, w.size() * w[0].size());
        d_C_T_buf = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_weight_size); CL_CHECK(err);

        // Allocate layer-specific buffers
        for (int i = 0; i < layers; ++i) {
            size_t c_size = cweights[i].size() * cweights[i][0].size();
            size_t b_size = bweights[i].size() * bweights[i][0].size();
            size_t act_size = activate[i].size();
            d_cweights[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, c_size * sizeof(float)); CL_CHECK(err);
            d_bweights[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, b_size * sizeof(float)); CL_CHECK(err);
            d_gradC[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, c_size * sizeof(float)); CL_CHECK(err);
            d_gradB[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, b_size * sizeof(float)); CL_CHECK(err);
            d_dotProds[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
            d_activate[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
            d_incoming[i] = cl::Buffer(clContext, CL_MEM_READ_WRITE, act_size * sizeof(float)); CL_CHECK(err);
        }

        if (layers > 1) {
            d_preoutgoing_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width); CL_CHECK(err);
            d_outgoing_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width); CL_CHECK(err);
            d_dpow_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width); CL_CHECK(err);
            d_dact_l = cl::Buffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * max_layer_width); CL_CHECK(err);
        }

        int epoch = 0;
        float initialLR = this->learningRate;
        int maxEpochs = 2000;

        // --- Training Loop ---
        while (epoch < maxEpochs) {
            // Copy weights H2D for current iteration
            for (int i = 0; i < layers; ++i) {
                std::vector<float> flat_c = flatten(cweights[i]);
                std::vector<float> flat_b = flatten(bweights[i]);
                CL_CHECK(clCommandQueue.enqueueWriteBuffer(d_cweights[i], CL_TRUE, 0, flat_c.size() * sizeof(float), flat_c.data()));
                CL_CHECK(clCommandQueue.enqueueWriteBuffer(d_bweights[i], CL_TRUE, 0, flat_b.size() * sizeof(float), flat_b.data()));
            }

            // --- Forward Propagation (adapted from mnn1d::clForprop) ---
            cl::Buffer d_current_act = d_in;
            cl::Kernel kernelForward = kernels.at("kernelLayerForward2");
            cl::Kernel kernelSigmoid = kernels.at("sigmoid");

            // First layer
            int current_in_size = input.size();
            int current_out_size = width[0];
            kernelForward.setArg(0, d_current_act);
            kernelForward.setArg(1, d_dotProds[0]);
            kernelForward.setArg(2, d_cweights[0]);
            kernelForward.setArg(3, d_bweights[0]);
            kernelForward.setArg(4, current_in_size);
            kernelForward.setArg(5, current_out_size);
            kernelForward.setArg(6, order);
            cl::NDRange globalForward = calculate_global_1d(WORKSIZE_1D, current_out_size);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForward, cl::NullRange, globalForward, local_1d));

            kernelSigmoid.setArg(0, d_dotProds[0]);
            kernelSigmoid.setArg(1, d_activate[0]);
            kernelSigmoid.setArg(2, current_out_size);
            cl::NDRange globalSig = calculate_global_1d(WORKSIZE_1D, current_out_size);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSigmoid, cl::NullRange, globalSig, local_1d));

            // Subsequent layers
            for (int i = 1; i < layers; ++i) {
                d_current_act = d_activate[i - 1];
                current_in_size = width[i - 1];
                current_out_size = width[i];
                kernelForward.setArg(0, d_current_act);
                kernelForward.setArg(1, d_dotProds[i]);
                kernelForward.setArg(2, d_cweights[i]);
                kernelForward.setArg(3, d_bweights[i]);
                kernelForward.setArg(4, current_in_size);
                kernelForward.setArg(5, current_out_size);
                kernelForward.setArg(6, order);
                globalForward = calculate_global_1d(WORKSIZE_1D, current_out_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelForward, cl::NullRange, globalForward, local_1d));

                kernelSigmoid.setArg(0, d_dotProds[i]);
                kernelSigmoid.setArg(1, d_activate[i]);
                kernelSigmoid.setArg(2, current_out_size);
                globalSig = calculate_global_1d(WORKSIZE_1D, current_out_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSigmoid, cl::NullRange, globalSig, local_1d));
            }
            CL_CHECK(clCommandQueue.finish());

            // Copy output D2H to check for correctness and loss
            CL_CHECK(clCommandQueue.enqueueReadBuffer(d_activate[layers - 1], CL_TRUE, 0, output.size() * sizeof(float), output.data()));
            output = softmax(output, SOFTMAX_TEMP);

            if (maxIndex(output) == maxIndex(target)) {
                std::cout << "Correct output predicted at epoch " << epoch << " with loss " << crossEntropy(output, target) << "." << std::endl;
                break;
            }
            epoch++;
            currloss = crossEntropy(output, target);
            std::cout << "Current CE Loss at epoch " << epoch << " : " << currloss << std::endl;

            // --- Backward Propagation (adapted from mnn1d::clBackprop) ---
            cl::Kernel kernelSub = kernels.at("subtract");
            cl::Kernel kernelSigmoidDer = kernels.at("sigmoidDer");
            cl::Kernel kernelScale = kernels.at("scaleByValue");
            cl::Kernel kernelDPow = kernels.at("dPower");
            cl::Kernel kernelVecxVec2Mat = kernels.at("vecxvec2mat");
            cl::Kernel kernelVecxMat2Vec = kernels.at("vecxmat2vec");
            cl::Kernel kernelTranspose = kernels.at("transpose");
            cl::Kernel kernelHadamard = kernels.at("hadamard2");
            
            cl::Kernel kernelUpdateWeights;
            switch (weightUpdateType) {
                case 0: kernelUpdateWeights = kernels.at("kernelUpdateWeights"); break;
                case 1: kernelUpdateWeights = kernels.at("kernelUpdateWeightsWithL1"); break;
                case 2: kernelUpdateWeights = kernels.at("kernelUpdateWeightsWithL2"); break;
                case 3: kernelUpdateWeights = kernels.at("kernelUpdateWeightsElasticNet"); break;
                case 4: kernelUpdateWeights = kernels.at("kernelUpdateWeightsWithWeightDecay"); break;
                case 5: kernelUpdateWeights = kernels.at("kernelUpdateWeightsDropout"); break;
                default: throw std::runtime_error("Invalid weight update type");
            }

            // Calculate initial error (output - expected)
            kernelSub.setArg(0, d_out);
            kernelSub.setArg(1, d_exp);
            kernelSub.setArg(2, d_err);
            kernelSub.setArg(3, (int)output.size());
            cl::NDRange globalSub = calculate_global_1d(WORKSIZE_1D, output.size());
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSub, cl::NullRange, globalSub, local_1d));
            CL_CHECK(clCommandQueue.enqueueCopyBuffer(d_err, d_incoming[layers - 1], 0, 0, sizeof(float) * output.size()));

            // Backpropagation loop (last layer to second layer)
            for (int layer = layers - 1; layer >= 1; --layer) {
                int prev_size = activate[layer - 1].size();
                int curr_size = activate[layer].size();
                int cweight_rows = prev_size;
                int cweight_cols = curr_size;
                size_t cweight_flat_size = cweight_rows * cweight_cols;

                cl::NDRange globalWeightGrad = calculate_global_1d(WORKSIZE_1D, cweight_flat_size);
                cl::NDRange globalOutGrad = calculate_global_1d(WORKSIZE_1D, prev_size);

                // dL/dC_l (Outer Product: d_activate[L-1] x d_incoming[L])
                kernelVecxVec2Mat.setArg(0, d_activate[layer - 1]);
                kernelVecxVec2Mat.setArg(1, d_incoming[layer]);
                kernelVecxVec2Mat.setArg(2, d_gradC[layer]);
                kernelVecxVec2Mat.setArg(3, cweight_rows);
                kernelVecxVec2Mat.setArg(4, cweight_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelVecxVec2Mat, cl::NullRange, globalWeightGrad, local_1d));

                // scale gradc by ALPHA
                kernelScale.setArg(0, d_gradC[layer]);
                kernelScale.setArg(1, d_gradC[layer]);
                kernelScale.setArg(2, ALPHA);
                kernelScale.setArg(3, (int)cweight_flat_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGrad, local_1d));

                // dL/dB_l (Outer Product: ones x d_incoming[L])
                kernelVecxVec2Mat.setArg(0, d_ones);
                kernelVecxVec2Mat.setArg(1, d_incoming[layer]);
                kernelVecxVec2Mat.setArg(2, d_gradB[layer]);
                kernelVecxVec2Mat.setArg(3, cweight_rows);
                kernelVecxVec2Mat.setArg(4, cweight_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelVecxVec2Mat, cl::NullRange, globalWeightGrad, local_1d));

                // scale gradb by 1-ALPHA
                kernelScale.setArg(0, d_gradB[layer]);
                kernelScale.setArg(1, d_gradB[layer]);
                kernelScale.setArg(2, 1.0f - ALPHA);
                kernelScale.setArg(3, (int)cweight_flat_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGrad, local_1d));

                // --- Outgoing Gradient Calculation (for layer-1) ---
                kernelTranspose.setArg(0, d_cweights[layer]);
                kernelTranspose.setArg(1, d_C_T_buf);
                kernelTranspose.setArg(2, cweight_rows);
                kernelTranspose.setArg(3, cweight_cols);
                cl::NDRange globalTranspose = calculate_global_2d(size2d, cweight_rows, cweight_cols);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelTranspose, cl::NullRange, globalTranspose, local_2d));

                // incoming gradient x C^T
                kernelVecxMat2Vec.setArg(0, d_incoming[layer]);
                kernelVecxMat2Vec.setArg(1, d_C_T_buf);
                kernelVecxMat2Vec.setArg(2, d_preoutgoing_l);
                kernelVecxMat2Vec.setArg(3, cweight_cols); // matRows
                kernelVecxMat2Vec.setArg(4, cweight_rows); // matCols
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelVecxMat2Vec, cl::NullRange, globalOutGrad, local_1d));

                // derivative of power
                kernelDPow.setArg(0, d_activate[layer - 1]);
                kernelDPow.setArg(1, d_dpow_l);
                kernelDPow.setArg(2, order);
                kernelDPow.setArg(3, prev_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelDPow, cl::NullRange, globalOutGrad, local_1d));

                // derivative of activation
                kernelSigmoidDer.setArg(0, d_dotProds[layer - 1]);
                kernelSigmoidDer.setArg(1, d_dact_l);
                kernelSigmoidDer.setArg(2, prev_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelSigmoidDer, cl::NullRange, globalOutGrad, local_1d));

                // outgoing gradient = (dl/dz_l x C^T) . dprev_p . dprevAct
                kernelHadamard.setArg(0, d_preoutgoing_l);
                kernelHadamard.setArg(1, d_dpow_l);
                kernelHadamard.setArg(2, d_dact_l);
                kernelHadamard.setArg(3, d_incoming[layer - 1]);
                kernelHadamard.setArg(4, 1);
                kernelHadamard.setArg(5, prev_size);
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelHadamard, cl::NullRange, globalOutGrad, local_1d));
            }

            // Backpropagation for the first layer (input layer)
            int prev_size = input.size();
            int curr_size = activate[0].size();
            int cweight_rows = prev_size;
            int cweight_cols = curr_size;
            size_t cweight_flat_size = cweight_rows * cweight_cols;
            cl::NDRange globalWeightGradFirst = calculate_global_1d(WORKSIZE_1D, cweight_flat_size);

            // dL/dC_0 (Outer Product: d_in x d_incoming[0])
            kernelVecxVec2Mat.setArg(0, d_in);
            kernelVecxVec2Mat.setArg(1, d_incoming[0]);
            kernelVecxVec2Mat.setArg(2, d_gradC[0]);
            kernelVecxVec2Mat.setArg(3, cweight_rows);
            kernelVecxVec2Mat.setArg(4, cweight_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelVecxVec2Mat, cl::NullRange, globalWeightGradFirst, local_1d));

            // scale gradc by ALPHA
            kernelScale.setArg(0, d_gradC[0]);
            kernelScale.setArg(1, d_gradC[0]);
            kernelScale.setArg(2, ALPHA);
            kernelScale.setArg(3, (int)cweight_flat_size);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGradFirst, local_1d));

            // dL/dB_0 (Outer Product: ones x d_incoming[0])
            kernelVecxVec2Mat.setArg(0, d_ones);
            kernelVecxVec2Mat.setArg(1, d_incoming[0]);
            kernelVecxVec2Mat.setArg(2, d_gradB[0]);
            kernelVecxVec2Mat.setArg(3, cweight_rows);
            kernelVecxVec2Mat.setArg(4, cweight_cols);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelVecxVec2Mat, cl::NullRange, globalWeightGradFirst, local_1d));

            // scale gradb by 1-ALPHA
            kernelScale.setArg(0, d_gradB[0]);
            kernelScale.setArg(1, d_gradB[0]);
            kernelScale.setArg(2, 1.0f - ALPHA);
            kernelScale.setArg(3, (int)cweight_flat_size);
            CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelScale, cl::NullRange, globalWeightGradFirst, local_1d));
            CL_CHECK(clCommandQueue.finish());

            // Update weights and copy to host vectors
            for (int i = 0; i < this->layers; ++i) {
                size_t c_size = cweights[i].size() * cweights[i][0].size();
                size_t b_size = bweights[i].size() * bweights[i][0].size();
                cl::NDRange globalUpdate = calculate_global_1d(WORKSIZE_1D, c_size);

                // Update C weights using kernelUpdateWeights
                kernelUpdateWeights.setArg(0, d_cweights[i]);
                kernelUpdateWeights.setArg(1, d_gradC[i]);
                switch (weightUpdateType) {
                    case 0:
                        kernelUpdateWeights.setArg(2, learningRate);
                        kernelUpdateWeights.setArg(3, (int)c_size);
                        break;
                    case 1:
                        kernelUpdateWeights.setArg(2, (int)c_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, LAMBDA_L1);
                        break;
                    case 2:
                        kernelUpdateWeights.setArg(2, (int)c_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, LAMBDA_L2);
                        break;
                    case 3:
                        kernelUpdateWeights.setArg(2, (int)c_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, LAMBDA_L1);
                        kernelUpdateWeights.setArg(5, LAMBDA_L2);
                        break;
                    case 4:
                        kernelUpdateWeights.setArg(2, (int)c_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, WEIGHT_DECAY);
                        break;
                    case 5:
                        kernelUpdateWeights.setArg(2, (int)c_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, DROPOUT_RATE);
                        kernelUpdateWeights.setArg(5, (uint)rand());
                        break;
                }
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdateWeights, cl::NullRange, globalUpdate, local_1d));
                // Update B weights using kernelUpdateWeights
                kernelUpdateWeights.setArg(0, d_bweights[i]);
                kernelUpdateWeights.setArg(1, d_gradB[i]);
                switch (weightUpdateType) {
                    case 0:
                        kernelUpdateWeights.setArg(2, learningRate);
                        kernelUpdateWeights.setArg(3, (int)b_size);
                        break;
                    case 1:
                        kernelUpdateWeights.setArg(2, (int)b_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, LAMBDA_L1);
                        break;
                    case 2:
                        kernelUpdateWeights.setArg(2, (int)b_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, LAMBDA_L2);
                        break;
                    case 3:
                        kernelUpdateWeights.setArg(2, (int)b_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, LAMBDA_L1);
                        kernelUpdateWeights.setArg(5, LAMBDA_L2);
                        break;
                    case 4:
                        kernelUpdateWeights.setArg(2, (int)b_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, WEIGHT_DECAY);
                        break;
                    case 5:
                        kernelUpdateWeights.setArg(2, (int)b_size);
                        kernelUpdateWeights.setArg(3, learningRate);
                        kernelUpdateWeights.setArg(4, DROPOUT_RATE);
                        kernelUpdateWeights.setArg(5, (uint)rand());
                        break;
                }
                CL_CHECK(clCommandQueue.enqueueNDRangeKernel(kernelUpdateWeights, cl::NullRange, globalUpdate, local_1d));

                // copy and reshape
                std::vector<float> c_upd(c_size), b_upd(b_size);
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_cweights[i], CL_TRUE, 0, sizeof(float) * c_size, (void*)c_upd.data()));
                CL_CHECK(clCommandQueue.enqueueReadBuffer(d_bweights[i], CL_TRUE, 0, sizeof(float) * b_size, (void*)b_upd.data()));
                cweights[i] = reshape(c_upd, cweights[i].size(), cweights[i][0].size());
                bweights[i] = reshape(b_upd, bweights[i].size(), bweights[i][0].size());
            }
        }

        this->learningRate = initialLR;
        if (epoch == maxEpochs) {
            std::cout << "Max epochs reached without convergence." << std::endl;
        }
        std::cout << "Training complete for this input-target pair." << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "Error during clBufTrain: " << e.what() << std::endl;
    }

    // --- Buffer Cleanup ---
    d_in = cl::Buffer(); d_exp = cl::Buffer(); d_out = cl::Buffer(); d_err = cl::Buffer(); d_ones = cl::Buffer();
    d_preoutgoing_l = cl::Buffer(); d_outgoing_l = cl::Buffer(); d_dpow_l = cl::Buffer(); d_dact_l = cl::Buffer(); d_C_T_buf = cl::Buffer();
    for (int i = 0; i < layers; ++i) {
        d_cweights[i] = cl::Buffer(); d_bweights[i] = cl::Buffer();
        d_gradC[i] = cl::Buffer(); d_gradB[i] = cl::Buffer();
        d_dotProds[i] = cl::Buffer(); d_activate[i] = cl::Buffer();
        d_incoming[i] = cl::Buffer();
    }
}

#endif // USE_CL