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

- The core logic is planned within the `src` directory. The implementation defines two main classes in `src/mnn.hpp`:
  - **`mnn`**: A class designed to represent a Monomial Neural Network for 1-dimensional input and output data (vectors).
  - **`mnn2d`**: A class designed to represent a Monomial Neural Network for 2-dimensional input and output data (matrices/images).
  - Headers for loss, activations and class definitions are provided separately.
  - Source files are provided for functions of specific purpose.
- `src` directory has `C++`, `OpenCL` and `CUDA` support subdirectory.
- Certain common functions are commonly used in all code for data access and modification.
- Binary files are used for weights storage (serialisation and deserialisation).

### Backend Support
- The project is configurable to run on different hardware backends.
  - **`USE_CUDA`**: Enables GPU acceleration using NVIDIA's CUDA.
  - **`USE_CL`**: Enables GPU/accelerator support via OpenCL, for broader hardware compatibility.
  - **`USE_CPU`**: Defaults to standard C++ for execution on the CPU.
- The backend can be selected by setting the corresponding flag in the `CMakeLists.txt` file.

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
- This is an assumption, since i have not included application of activation on monomial and its deriavative in update of coefficient.
  - Let a(.) be activation functon and a'(.) be its derivative.
  - v = a(c(x^n) + b) is the output of the monomial neuron after activation.
  - To properly update the coefficients `c` and `b` when an activation function is involved, we must use the chain rule from calculus. The goal is to find the partial derivative of the Loss function `E` with respect to `c` and `b`.
  - Let `z = c(x^n) + b` be the pre-activation output. The output of the neuron is `v = a(z)`.
  - The gradients for `c` and `b` are calculated as follows:
    - `∂E/∂c = ∂E/∂v * ∂v/∂z * ∂z/∂c`
    - `∂E/∂b = ∂E/∂v * ∂v/∂z * ∂z/∂b`
  - Where:
    - `∂E/∂v` is the error signal propagated backward from the next layer.
    - `∂v/∂z = a'(z) = a'(c(x^n) + b)` is the derivative of the activation function.
    - `∂z/∂c = x^n`
    - `∂z/∂b = 1`
  - Substituting these in, we get the gradients:
    - **Gradient for c**: `∂E/∂c = (∂E/∂v) * a'(c(x^n) + b) * (x^n)`
    - **Gradient for b**: `∂E/∂b = (∂E/∂v) * a'(c(x^n) + b)`
  - The final update rules using the learning rate `L` are:
    - `c <- c - L * (∂E/∂v) * a'(c(x^n) + b) * (x^n)`
    - `b <- b - L * (∂E/∂v) * a'(c(x^n) + b)`

## Theorem Sketch
- _**Universal Approximation (given by Grok)**_: Monomial networks with fixed $ m \geq 2 $ and sufficient layers/neurons can approximate any continuous function on compact sets. Why?
  - Powers `x^m` span nonlinear basis
  - Layer composition generates dense function class
  - More expressive than linear MLPs for same width/depth
