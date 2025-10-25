cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)

include_directories(include)

add_library(mnn STATIC
    activations.cpp
    loss.cpp
    operators.cpp
    weights.cpp

    mnn.cpp
    mnn2d.cpp
    layer.cpp
    forprop.cpp
    backprop.cpp
    train.cpp
)