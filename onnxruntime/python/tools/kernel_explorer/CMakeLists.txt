# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

project(kernel_explorer)
cmake_minimum_required(VERSION 3.21)

include(${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/external/pybind11.cmake)

set(CMAKE_CXX_COMPILER ${onnxruntime_ROCM_HOME}/bin/hipcc)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_PREFIX_PATH ${onnxruntime_ROCM_HOME} ${onnxruntime_ROCM_HOME}/hip)
list(APPEND CMAKE_MODULE_PATH ${onnxruntime_ROCM_HOME}/hip/cmake)
find_package(HIP REQUIRED)
find_package(rocblas REQUIRED)
find_package(Python COMPONENTS Development REQUIRED)

include_directories(${Python_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(.)

file(GLOB kernel_srcs kernels/*.cpp CONFIG)
add_library(kernel_explorer SHARED kernel_explorer.cpp timer.cpp hip_utils.cpp ${kernel_srcs})
set_target_properties(kernel_explorer PROPERTIES PREFIX "")
target_link_libraries(kernel_explorer PRIVATE ${PYTHON_LIBRARIES} roc::rocblas)
target_compile_definitions(kernel_explorer PUBLIC ROCM_USE_FLOAT16)

enable_testing()
find_package(Python COMPONENTS Interpreter REQUIRED)
add_test(NAME test_kernels COMMAND ${Python_EXECUTABLE} -m pytest ..)
