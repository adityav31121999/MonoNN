# MonoNN
- This is an experimental project to study the modification to multi-layer perception with linear to monomial based neurons.
- This modification still employs basic structure of MLP but with monomial neurons.
- The monomial is of the form f(x) = c*(x^n) + b
  - x: input to monimial
  - n: order of all monomial, neurons and mlp
  - c: coefficient of x^n
  - b: constant
- This modification is done to understand the nature of non-linearity over linear nature of mlp.
- This helps in study of direct impact of non-linearity to results of mlp and optimisation of weights and how much variation compared to standard mlp.
