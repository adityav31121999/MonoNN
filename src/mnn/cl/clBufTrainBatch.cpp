#ifdef USE_OPENCL
#include "mnn.hpp"
#include <vector>
#include <stdexcept>

void mnn::clBufTrainBatch(const std::vector<float>& in, const std::vector<float>& exp)
{
    // buffers for mlp

    // forprop

    // error

    // backprop
}


void mnn2d::clBufTrainBatch(const std::vector<std::vector<float>>& in, const std::vector<float>& exp)
{
    // buffers for mlp

    // forprop

    // error

    // backprop
}

#endif