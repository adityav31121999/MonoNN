# MonoNN: Monomial Neural Network (EXPERIMENTAL)

- This is an experimental project to study the modification to multi-layer perception from linear to monomial-based neurons.
- The monomial is of the form: `f(x) = c*(x^m) + b`
  - `x`: input to monimial
  - `m`: order of monomial, neurons and mlp
  - `c`: coefficient of $ x^m $
  - `b`: constant
  - Both c and b are trainable parameters.
- Here the `b` term is intentionally added to represent the `bias` as used in nerual neworks.
- This modification is done to understand the nature of non-linearity over linear nature of mlp, direct impact of non-linearity to results of mlp and optimisation of weights and how much variation compared to standard mlp.
- Derivative of Monomial is given as: `f'(x) = m*c*(x^(m-1))`

## Neural Network

- Two types of neural networks are defined based on input: MNN and MNN2D
  - MNN takes 1D vector input
  - MNN2D takes 2D matrix input
  - Both produce 1D output vector
- Both networks have similar mechanism and weight structure, with gradient storing per layer.
  - For 1D i/o, it has 1D product and Activations per layer.
  - For 2D i/o, it has 2D product and Activations per layer.
  - Output is calculated via Mean/Max/Weighted Mean Pooling in MNN2D.
  - 2D bias matrix for each weight matrix.
    - The reason behind this is simple, an MLP will have bias vector, and element wise added to the output obtained by product of previous activation and current weight.
    - Single bias value can tune the signal obtained by product of row and column vector. For monomial nature, this can be tricky, since it can explode or restrain the value.
    - Hence, each weight value has its own bias.
    - An alpha value (0.6) is utilised to make coefficients to absorb the major change, compared to biases that absorb the minor part.

## Gradients

Gradient calculation is done in following manner:

### _Gradients to update monomial:_

- Let a(.) be activation functon and a'(.) be its derivative.
  - v = a(c(x^m) + b) is the output of the monomial neuron after activation.
  - To properly update the coefficients `c` and `b` when an activation function is involved, we must use the chain rule from calculus. The goal is to find the partial derivative of the Loss function `E` with respect to `c` and `b`.
  - Let `z = c(x^m) + b` be the pre-activation output. The output of the neuron is `v = a(z)`.
  - The gradients for `c` and `b` are calculated as follows:
    - `∂E/∂c = ∂E/∂v * ∂v/∂z * ∂z/∂c`
    - `∂E/∂b = ∂E/∂v * ∂v/∂z * ∂z/∂b`
  - Where:
    - `∂E/∂v` is the error signal propagated backward from the next layer.
    - `∂v/∂z = a'(z) = a'(c(x^m) + b)` is the derivative of the activation function.
    - `∂z/∂c = x^m`
    - `∂z/∂b = 1`
  - Substituting these in, we get the gradients:
    - **Gradient for c**: `∂E/∂c = (∂E/∂v) * a'(c(x^m) + b) * (x^m)`
    - **Gradient for b**: `∂E/∂b = (∂E/∂v) * a'(c(x^m) + b)`
  - The final update rules using the learning rate `L` are:
    - `c <- c - L * (∂E/∂v) * a'(c(x^m) + b) * (x^m)`
    - `b <- b - L * (∂E/∂v) * a'(c(x^m) + b)`

### _Gradients for perceptron:_

- Consider a perceptron with two neurons as follows:

  ```
             ____                       ____
            |    |  z1                 |    | z2
    I ----> | M1 | ---> a(.) = a1 ---> | M2 | ---> a(.) = a2 ---> O
            |____|                     |____|

  ```

  - I and O indicate the input and output, with M1 and M2 being the monomial neurons, z1 and z2 are their respective output.
  - Hence the gradients for these are as follows, which in general is also similar to Larger networks:
    - `M1 = C1*(x^m) + B1, M2 = C2*(x^m) + B2`
    - Let L be the loss: `∂L/∂O = O - Expected`
    - For monomial M2:
      - Incoming gradient for M2: `∂L/∂z2 = ∂L/∂O * ∂O/∂z2`
      - Gradient for C2: `∂L/∂C2 = ∂L/∂z2 * ∂z2/∂C2`
      - Gradient for B2: `∂L/∂B2 = ∂L/∂z2 * ∂z2/∂B2`
      - Outgoing gradient: `∂L/∂z1 = ∂L/∂z2 * ∂z2/∂a1 * ∂a1/∂z1`
    - For Monomial M1:
      - Incoming gradient for M2: `dL/dz1 = dL/dz2 * dz2/da1 * da1/dz1`
      - Gradient for C1: `∂L/∂C1 = ∂L/∂z1 * ∂z1/∂C1`
      - Gradient for B1: `∂L/∂B1 = ∂L/∂z1 * ∂z1/∂B1`
    - `∂a1/∂z1` and `∂a2/∂z2` are derivative of the activation.

- **For batch of input and output on this perceptron, the gradients will be:**
  ```
      I(1)                                                            O(1)
      I(2)                                                            O(2)
      I(3)                                                            O(3)
       :         ____                       ____                       :
       :        |    |  z1                 |    | z2                   :
       :  ----> | M1 | ---> a(.) = a1 ---> | M2 | ---> a(.) = a2 --->  :
       :        |____|                     |____|                      :
       :                                                               :
       :                                                               :
       :                                                               :
      I(n)                                                            O(n)
  ```

  - `I(i)`: ith input, `O(i)`: ith output, `E(i)`: ith expected output
  - `z1(i)`: ith input's M1 output, `z2(i)`: ith input's M2 output
  - `a1(i)`: activation of z1(i), `a2(i)`: activation of z2(i)
  - `L(i)`: Loss recorded for ith output
  - Gradient of Error for ith will be: `∂L(i)/∂O(i) = O(i) - E(i)`
  - 

## Gradients for Network

- Similar to perceptron, the whole mechanism follows similar math, though for `MNN` and `MNN2D` it is different.
- Though the process is similar, in `MNN` the gradients are passed as vectors and in `MNN2D` they are passes matrices.
- Final results are almost similar in notation.
- Loss for both is calculated from cross entropy: 
  - For MNN: `CE = -sum(T(i)logP(i))`
  - For MNN2D: `CE = -sum(T(i, j)logP(i, j))`
  - `T`: Target (expected output from neural network)
  - `P`: Prediction (output of neural network)

### _Gradients for MNN:_
- **For Sigle Input:**
  - Lets say there are `l` layers of hidden weights.
  - The gradient from loss w.r.t. `∂L/∂zl = ∂(CE)/∂zl = P - T = δ(l)` (element-wise subtraction)
  - This loss gradient is used to backpropagate till first layer for both coefficients and weights.
  - From last to second layer:
    - gradient for layer `i-1`: `∂L/∂z(i-1) = ∂L/∂zi * ∂zi/∂a(i-1) * ∂a(i-1)/∂z(i-1)`
    - `δ(i-1) = ∂L/∂z(i-1) = (∂L/∂zi * Ci^T) . (m * (a(i-1)^(m-1))) . (a'(i-1))`
    - `δ(i)` is used to update Cl and Bl weights
  - For all layer weights, C and B:
    - `∂L/∂Bi = ∂L/∂zi * ∂zi/∂Bi`, and `∂L/∂Ci = ∂L/∂zi * ∂zi/∂Bi`
    - For C: `∂L/∂Ci = p(l-1) * ∂L/∂zi`, here `p(l-1) = a(l-1) ^ m`
    - For B: `∂L/∂Bi = v1 * ∂L/∂zi`, here `v1` is the column vector of 1s with same size of a(i-1)

- **For Batch Input:**
  - 

### _Gradients for MNN2D:_
- Here also, the process and notations are pretty similar.
- **For Sigle Input:**
  - The gradient from loss w.r.t. `∂L/∂zl = ∂(CE)/∂zl = P - T = δ(l)` (element-wise subtraction)
  - This loss gradient is used to backpropagate till first layer for both coefficients and weights.
  - From last to second layer:
    - gradient for layer `i-1`: `∂L/∂z(i-1) = ∂L/∂zi * ∂zi/∂a(i-1) * ∂a(i-1)/∂z(i-1)`
    - `δ(i-1) = ∂L/∂z(i-1) = (∂L/∂zi * Ci^T) . (m * (a(i-1)^(m-1))) . (a'(i-1))`
    - `δ(i)` is used to update Cl and Bl weights
  - For all layer weights, C and B:
    - `∂L/∂Bi = ∂L/∂zi * ∂zi/∂Bi`, and `∂L/∂Ci = ∂L/∂zi * ∂zi/∂Bi`
    - For C: `∂L/∂Ci = (p(l-1)^T) * ∂L/∂zi`, here `p(l-1) = a(l-1) ^ m`
    - For B: `∂L/∂Bi = v1 * ∂L/∂zi`, here `v1` is the matrix of 1s with dimension {rows of Ci x rows of input matrix}

- **For Batch Input:**
  - 

---

## Project Versions

- **0.0.1**: Basic Structure and File-by-File Training (complete)
  - Neural Network Structure
  - Calculation for single input training
  - Backend support for compute
- **0.0.2**: Batch Training (in progress)
  - Calculation for batch input training

## Project Structure

- `src/operators.hpp` is provided to have C++, CUDA and OpenCL support.
  - Hyperparameters such as learning rate, decay rate and regularisation parameters are provided.
  - `OpenCL` context function is declared alongwith error support.
  - `Cuda` kernel declarations are provided.
  - `C++` operations and functions are declared here too.
- The core logic is planned within the `src` directory. The implementation defines two main classes in `src/mnn.hpp` and `src/mnn2.hpp`:
  - **`mnn`**: A class designed to represent a Monomial Neural Network for 1-dimensional input and output data (vectors).
  - **`mnn2d`**: A class designed to represent a Monomial Neural Network for 2-dimensional input and output data (matrices/images).
  - Headers for loss, activations and class definitions are provided separately.
  - Source files are provided for functions of specific purpose.
- `src` directory has `C++`, `OpenCL` and `CUDA` support subdirectory.
- Certain common functions are commonly used in all code for data access and modification.
- Binary files are used for weights storage (serialisation and deserialisation).

## Features

The library is built with a modular approach, separating functionalities into different files.

### _Backend Support_

- The project is configurable to run on different hardware backends.
  - `USE_CU`: Enables GPU acceleration using NVIDIA's CUDA.
  - `USE_CL`: Enables GPU/accelerator support via OpenCL, for broader hardware compatibility.
  - `USE_CPU`: Defaults to standard C++ for execution on the CPU.
- The backend can be selected by setting the corresponding flag in the `CMakeLists.txt` file.

### _Functions_

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
- **Batch Training**: Support for training on batches of data.
- **Discrete File Training**: Support for training on discrete files.

### _Activation Functions_

- **Sigmoid**: `sigmoid` function and its derivative (for `MNN`)
- **ReLU**: Rectified Linear Unit (`relu`) and its derivative (Not Utilised)
- **Softmax**: `softmax` function with optional temperature parameter for probability distribution (for `MNN2D`)

### _Loss Functions_

- **MSE**: Mean Squared Error (`mse`) for regression tasks.
- **Cross Entropy**: Standard Cross Entropy (`crossEntropy`) for classification.
- **Binary Cross Entropy**: `binaryCrossEntropy` for binary classification tasks.
- **Categorical Cross Entropy**: `categoricalCrossEntropy` for multi-class classification with 2D outputs.

### _Image Processing_

- **OpenCV Integration**:
  - `image2grey`: Loads an image as a grayscale matrix.
  - `image2channels`: Loads an image and splits it into RGB channels.
  - `cvMat2vec` / `vec2cvMat`: Converters between OpenCV `cv::Mat` and `std::vector` formats.

### _Utilities_

- **Serialization**:
  - Fixed number of values for saving and loading weights (`serializeWeights`, `deserializeWeights`).
- **Progress Tracking**:
  - `progress` struct tracks training metrics (loss, accuracy, time, cycles).
  - CSV logging support via `logProgressToCSV`.
- **Statistics**:
  - `computeStats`: Calculates mean, standard deviation, min, and max for data analysis.

### _Default Hyperparameters_

- Defined in `operators.hpp`, these serve as default configuration values:
  - `LEARNING_MAX`: 0.01
  - `LEARNING_MIN`: 0.00001
  - `BATCH_SIZE`: 50
  - `EPOCH`: 100
  - `DROPOUT_RATE`: 0.6
  - `LAMBDA_L1` / `LAMBDA_L2`: 0.001
  - `SOFTMAX_TEMP`: 1.5


## Theorem Sketch

### **Universal Approximation (generated by Grok)**

- Monomial networks with fixed `m >= 2` and sufficient layers/neurons can approximate any continuous function on compact sets. Why?
  - Powers `x^m` span nonlinear basis
  - Layer composition generates dense function class
  - More expressive than linear MLPs for same width/depth
