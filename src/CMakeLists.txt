cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
include_directories(include)

set(MNN_SOURCES
    mnn/cpp/operators.cpp
    mnn/activations.cpp
    mnn/loss.cpp
    mnn/weights.cpp
    mnn/mnn.cpp
    mnn/mnn2d.cpp
    mnn/access.cpp
    mnn/image.cpp
    mnn/train.cpp
    mnn/test.cpp
    mnn/stats.cpp
)

if(USE_CPU)
    list(APPEND MNN_SOURCES
        mnn/cpp/layer.cpp
        mnn/cpp/forprop.cpp
        mnn/cpp/backprop.cpp
        mnn/cpp/train.cpp
    )
elseif(USE_CU)
    list(APPEND MNN_SOURCES
        mnn/cu/device_functions.cuh
        mnn/cu/activations.cu
        mnn/cu/operators.cu
        mnn/cu/forward.cu
        mnn/cu/forprop.cu
        mnn/cu/forpropb.cu
        mnn/cu/backprop.cu
        mnn/cu/backBatch.cu
        mnn/cu/train.cu
    )
elseif(USE_CL)
    list(APPEND MNN_SOURCES
        mnn/cl/activations.cl
        mnn/cl/operators.cl
        mnn/cl/forprop.cl
        mnn/cl/kernels.cpp
        mnn/cl/clforprop.cpp
        mnn/cl/clforpropb.cpp
        mnn/cl/clbackprop.cpp
        mnn/cl/clBackBatch.cpp
        mnn/cl/cltrain.cpp
    )
endif()

add_library(mnn STATIC ${MNN_SOURCES})
target_link_libraries(mnn
    PUBLIC # Important: Make OpenCL and CUDA linking public
        ${OpenCL_LIBRARIES}
        ${CUDAToolkit_LIBRARIES}
        ${OpenCV_LIBRARIES}
)