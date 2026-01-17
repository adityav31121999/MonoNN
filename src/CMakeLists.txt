cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
include_directories(include)

set(MNN_SOURCES
    cppfunc/operators.cpp
    cppfunc/activations.cpp
    cppfunc/loss.cpp
    cppfunc/updateweights.cpp
    cppfunc/setweights.cpp
    cppfunc/schedLr.cpp
    files/accessbin.cpp
    files/image.cpp
    evaluation/stats.cpp
    evaluation/session.cpp
    evaluation/progress.cpp
    evaluation/confusion.cpp
    evaluation/epoch.cpp

    nets/mnn1d/mnn1d.cpp
    nets/mnn2d/mnn2d.cpp

    nets/mnn1d/trainNtest/preTrain.cpp # MNN1D
    nets/mnn1d/trainNtest/onlineTrain.cpp # MNN1D
    nets/mnn1d/trainNtest/batchTrain.cpp # MNN1D
    nets/mnn1d/trainNtest/fullsetTrain.cpp # MNN1D
    nets/mnn1d/trainNtest/test.cpp # MNN1D
    nets/mnn1d/trainNtest/testOnAll.cpp # MNN1D
    nets/mnn1d/trainNtest/trainNtest.cpp # MNN1D
    nets/mnn1d/trainNtest/postTrain.cpp # MNN1D

    nets/mnn2d/trainNtest/preTrain.cpp # MNN2D
    nets/mnn2d/trainNtest/onlineTrain.cpp # MNN2D
    nets/mnn2d/trainNtest/batchTrain.cpp # MNN2D
    nets/mnn2d/trainNtest/fullsetTrain.cpp # MNN2D
    nets/mnn2d/trainNtest/test.cpp # MNN2D
    nets/mnn2d/trainNtest/testOnAll.cpp # MNN2D
    nets/mnn2d/trainNtest/trainNtest.cpp # MNN2D
    nets/mnn2d/trainNtest/postTrain.cpp # MNN2D
)

if(USE_CL)
    list(APPEND MNN_SOURCES
        clkernel/activations.cl
        clkernel/forprop.cl
        clkernel/operators.cl
        clkernel/kernels.cpp
    )
elseif(USE_CU)
    list(APPEND MNN_SOURCES
        cukernel/activations.cu
        cukernel/forprop.cu
        cukernel/operators.cu
    )
elseif(USE_CPU)
    list(APPEND MNN_SOURCES
        cppfunc/multiply.cpp
        cppfunc/forproplayer.cpp
        cppfunc/backSingle.cpp
        cppfunc/backBatch.cpp
    )
endif() # End of USE_CL/USE_CU/USE_CPU for general kernels

#### MNN2D ####

if(USE_CPU)
    list(APPEND MNN_SOURCES
        nets/mnn1d/cpp/forprop.cpp
        nets/mnn1d/cpp/backprop.cpp
        nets/mnn1d/cpp/train.cpp
        nets/mnn1d/cpp/train1c.cpp
        nets/mnn1d/cpp/trainBatch1c.cpp
        nets/mnn1d/cpp/trainOnThreads.cpp
    )
elseif(USE_CU)
    list(APPEND MNN_SOURCES
        nets/mnn1d/cu/forprop.cu
        nets/mnn1d/cu/forpropb.cu
        nets/mnn1d/cu/backprop.cu
        nets/mnn1d/cu/backBatch.cu
        nets/mnn1d/cu/train.cu
        nets/mnn1d/cu/train1c.cu
        nets/mnn1d/cu/trainBatch1c.cu
        nets/mnn1d/cu/trainOnBuf1.cu
        nets/mnn1d/cu/trainOnBuf2.cu
    )
elseif(USE_CL)
    list(APPEND MNN_SOURCES
        nets/mnn1d/cl/clforprop.cpp
        nets/mnn1d/cl/clForpropb.cpp
        nets/mnn1d/cl/clbackprop.cpp
        nets/mnn1d/cl/clBackBatch.cpp
        nets/mnn1d/cl/cltrain.cpp
        nets/mnn1d/cl/clTrain1c.cpp
        nets/mnn1d/cl/clTrainBatch1c.cpp
        nets/mnn1d/cl/trainOnBuf1.cpp
        nets/mnn1d/cl/trainOnBuf2.cpp
    )
endif()

#### MNN2D ####

if(USE_CPU)
    list(APPEND MNN_SOURCES
        nets/mnn2d/cpp/forprop.cpp
        nets/mnn2d/cpp/backprop.cpp
        nets/mnn2d/cpp/train.cpp
        nets/mnn2d/cpp/train1c.cpp
        nets/mnn2d/cpp/trainBatch1c.cpp
        nets/mnn2d/cpp/trainOnThreads.cpp
    )
elseif(USE_CU)
    list(APPEND MNN_SOURCES
        nets/mnn2d/cu/forprop.cu
        nets/mnn2d/cu/forpropb.cu
        nets/mnn2d/cu/backprop.cu
        nets/mnn2d/cu/backBatch.cu
        nets/mnn2d/cu/train.cu
        nets/mnn2d/cu/train1c.cu
        nets/mnn2d/cu/trainBatch1c.cu
        nets/mnn2d/cu/trainOnBuf1.cu
        nets/mnn2d/cu/trainOnBuf2.cu
    )
elseif(USE_CL)
    list(APPEND MNN_SOURCES
        nets/mnn2d/cl/clforprop.cpp
        nets/mnn2d/cl/clForpropb.cpp
        nets/mnn2d/cl/clbackprop.cpp
        nets/mnn2d/cl/clBackBatch.cpp
        nets/mnn2d/cl/cltrain.cpp
        nets/mnn2d/cl/clTrain1c.cpp
        nets/mnn2d/cl/clTrainBatch1c.cpp
        nets/mnn2d/cl/trainOnBuf1.cpp
        nets/mnn2d/cl/trainOnBuf2.cpp
    )
endif()


add_library(mono STATIC ${MNN_SOURCES})
target_link_libraries(mono
    PUBLIC # Important: Make OpenCL and CUDA linking public
        ${OpenCL_LIBRARIES}
        ${CUDAToolkit_LIBRARIES}
        ${OpenCV_LIBRARIES}
)