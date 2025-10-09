cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)

include_directories(include)

add_library(mnn STATIC
    mnn.cpp
    mnn2d.cpp
    activations.cpp
)