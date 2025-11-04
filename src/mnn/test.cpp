#include "mnn.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>

void mnn::test(const std::string &dataSetPath, float loss)
{
    #ifdef USE_CPU
    #elif USE_CUDA
    #elif USE_OPENCL
    #endif
}