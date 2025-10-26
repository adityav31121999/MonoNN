cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)

include_directories(include)

set(MNN_SOURCES
    mnn/activations.cpp # Should be #include "activations.hpp"
    mnn/loss.cpp        # Should be #include "loss.hpp"
    mnn/operators.cpp   # etc.
    mnn/weights.cpp
    mnn/mnn.cpp
    mnn/mnn2d.cpp
    mnn/layer.cpp
)

if(USE_CPU)
    list(APPEND MNN_SOURCES
        mnn/cpp/forprop.cpp
        mnn/cpp/backprop.cpp
        mnn/cpp/train.cpp
    )
elseif(USE_CUDA)
    list(APPEND MNN_SOURCES
        mnn/cu/forprop.cu
        mnn/cu/backprop.cu
        mnn/cu/train.cu
    )
elseif(USE_OPENCL)
    list(APPEND MNN_SOURCES
        mnn/cl/clforprop.cpp
        mnn/cl/clbackprop.cpp
        mnn/cl/cltrain.cpp
    )
endif()

add_library(mnn STATIC ${MNN_SOURCES})