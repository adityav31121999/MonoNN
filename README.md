# MonoNN: Monomial Neural Network
- This is an experimental project to study the modification to multi-layer perception from linear to monomial-based neurons.
- The monomial is of the form: w => `f(x) = c*(x^n) + b`
  - `x`: input to monimial
  - `n`: order of monomial, neurons and mlp
  - `c`: coefficient of `x^n`
  - `b`: constant
  - Both c and b are trainable parameters.
- Here the `b` term is intentionally added to represent the `bias` as used in nerual neworks.
- This modification is done to understand the nature of non-linearity over linear nature of mlp, direct impact of non-linearity to results of mlp and optimisation of weights and how much variation compared to standard mlp.

## Project Structure

The core logic is planned within the `src` directory. The implementation defines two main classes in `src/mnn.hpp`:
  - **`mnn`**: A class designed to represent a Monomial Neural Network for 1-dimensional input and output data (vectors).
  - **`mnn2d`**: A class designed to represent a Monomial Neural Network for 2-dimensional input and output data (matrices/images).
  - Headers for loss, activations and class definitions are provided separately.
  - Source files are provided for functions of specific purpose.

### _MonoNN structure_
- Both MonoNN have similar mechanism and weight structure, with gradient storing also per hidden layer.
  - For 1D i/o, it has 1D product and Activations per layer.
  - For 2D i/o, it has 2D product and Activations per layer.
    - Output is calculated via Mean/Max/Weighted Mean Pooling.
  - Hyperparameters such as learning rate, decay rate and regularisation parameters are also provided.

## Features

The library is built with a modular approach, separating functionalities into different files.

### _operators.cpp and weights.cpp_
- **Matrix/Vector Operations**: Overloaded `operator*` for standard matrix-matrix and vector-matrix multiplication.
- **Multi-threaded Element-wise Multiplication**: Optimized `multiply` functions that leverage multi-threading for improved performance on modern CPUs.
- **Pooling Functions**:
  - `meanPool`: Computes the average pooling over matrix rows.
  - `maxPool`: Computes the max pooling over matrix rows.
  - `weightedMeanPool`: Computes a weighted average pooling.
- **Power Functions**: `power` function to apply element-wise exponentiation to vectors and matrices.
- **Weight Initialization**:
  - `setWeightsByNormalDist`: Initialize weights from a normal distribution.
  - `setWeightsByUniformDist`: Initialize weights from a uniform distribution.
  - `setWeightsByXavier`: Xavier/Glorot initialization.
  - `setWeightsByHe`: He initialization, suitable for ReLU activation functions.
  - `setWeightsByLeCunn`: LeCun initialization.
- **Weight Update Rules & Regularization**:
  - `updateWeights`: Standard gradient descent.
  - `updateWeightsL1`: Gradient descent with L1 regularization.
  - `updateWeightsL2`: Gradient descent with L2 regularization.
  - `updateWeightsElastic`: Gradient descent with Elastic Net (L1 & L2) regularization.
  - `updateWeightsWeightDecay`: Gradient descent with weight decay.
  - `updateWeightsDropout`: Applies dropout during weight updates.
- Similar to C++ operators, **OpenCL** and **CUDA** kernels are provided for fast training of MonoNNs.

### _Gradients_
- The gradient of Neurons are calculated as:
  - Derivative of Monomial is given as: g => `f'(x) = nc(x^(n-1))`
  - Parameter is updated as `w <- w - L.g`
    - `w` = weight to be updated
    - `L` = learning rate
    - `g` = gradient
    - `<-` signifies update applied to w
  - When the gradient is applied to, the `b` term will remain unaffected, hence a part of the gradient will be applied to it.
  - Hence: `w <- (c*(x^n) + b) - L.(nc(x^(n-1))) = c(1 - (L.n/x))(x^n) + b`
  - The term `1 - (L.n/x)` solely updates the `c` coefficient of monomial not the `b` term.
    - `c <- (1-(L.n/x)) * c`
  - So, a small part of it can be applied to the `b` term. So gradient can be split equally or specific value can be used to break and use it to update the coefficient with major part and minor part for `b`.
  - So the gradient term is `L(n-1)/x`, in this project will be split on the basis of `0.9 and 0.1` for both major and minor part.
    - `c <- 0.9 * (1 - (L.n/x)) * c`
    - `b <- 0.1 * (1 - (L.n/x)) * c`
  - Hence, based on this, the gradients that will calculated during backpropagation, will be split to c and b.
  - This helps in modifying the impact of both c and b.

## Theorem Sketch
- _**Universal Approximation (given by Grok)**_: Monomial networks with fixed $ m \geq 2 $ and sufficient layers/neurons can approximate any continuous function on compact sets. Why?
  - Powers $ \{x^m\} $ span nonlinear basis
  - Layer composition generates dense function class
  - More expressive than linear MLPs for same width/depth
