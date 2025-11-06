cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)

include_directories(include)

set(MNN_SOURCES
    mnn/activations.cpp # Should be #include "operators.hpp"
    mnn/loss.cpp        # Should be #include "loss.hpp"
    mnn/operators.cpp   # etc.
    mnn/weights.cpp
    mnn/mnn.cpp
    mnn/mnn2d.cpp
    mnn/access.cpp
    mnn/image.cpp
    mnn/train.cpp
    mnn/test.cpp
)

if(USE_CPU)
    list(APPEND MNN_SOURCES
        mnn/cpp/layer.cpp
        mnn/cpp/forprop.cpp
        mnn/cpp/backprop.cpp
        mnn/cpp/train.cpp
        mnn/cpp/threadTrain.cpp
        mnn/cpp/test.cpp
    )
elseif(USE_CUDA)
    list(APPEND MNN_SOURCES
        mnn/cu/kernel.cu
        mnn/cu/forprop.cu
        mnn/cu/backprop.cu
        mnn/cu/backBatch.cu
        mnn/cu/train.cu
        mnn/cu/bufTrain.cu
        mnn/cu/test.cu
    )
elseif(USE_OPENCL)
    list(APPEND MNN_SOURCES
        mnn/cl/kernel.cl
        mnn/cl/kernels.cpp
        mnn/cl/clforprop.cpp
        mnn/cl/clBackprop.cpp
        mnn/cl/clbackBatch.cpp
        mnn/cl/cltrain.cpp
        mnn/cl/clBufTrain.cpp
        mnn/cl/cltest.cpp
    )
endif()

add_library(mnn STATIC ${MNN_SOURCES})