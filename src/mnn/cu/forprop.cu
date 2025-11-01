#ifdef USE_CUDA
#include "include/mnn.hpp"
#include <cuda.h>

void mnn::cuForprop(std::vector<float> &input)
{
    float* cuInput, cuOutput;
    std::vector<float*> cw, bw;
    std::vector<float*> prod, act, power;

    try {
        //
    }
    catch () {
        // 
    }
}

#endif