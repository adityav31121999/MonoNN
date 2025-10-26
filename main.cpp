#include <iostream>
#include "mnn.h"

int main() {
    std::cout << "THIS IS MONOMIAL NEURAL NETWORK  IMPLEMENTATION" << std::endl;
    std::vector<float> hidden_layers = { 2, 4, 1, 4};
    std::vector<std::vector<float>> X = {
        {0, 0, 9},
        {0, 1, 7},
        {1, 0, 6},
        {1, 1, 2}
    };
    std::vector<float> Y(3, 0.0f);
    Y = hidden_layers * X;
    for (const auto& val : Y) {
        std::cout << val << " ";
    }

    return 0;
}