# MonoNN: Monomial Neural Network
- This is an experimental project to study the modification to multi-layer perception from linear to monomial-based neurons.
- The monomial is of the form f(x) = c*(x^n) + b
  - x: input to monimial
  - n: order of all monomial, neurons and mlp
  - c: coefficient of x^n
  - b: constant
  - Both c and b are trainable parameters.
- This modification is done to understand the nature of non-linearity over linear nature of mlp, direct impact of non-linearity to results of mlp and optimisation of weights and how much variation compared to standard mlp.

## Project Structure

-The core logic is planned within the `src` directory. The implementation defines two main classes in `src/mnn.hpp`:
  - **`mnn`**: A class designed to represent a Monomial Neural Network for 1-dimensional input and output data (vectors).
  - **`mnn2d`**: A class designed to represent a Monomial Neural Network for 2-dimensional input and output data (matrices/images).
- Headers for loss, activations and class definitions are provided separately.
- Source files are provided for functions of specific purpose.

## MonoNN structure
- Both MonoNN have similar mechanism and weight Structure just change in storing products and activations based on input-output dimensions.
  - For 1D i/o, it has 1D product and Activition per layer.
  - For 2D i/o, it has 2D product and activations.
