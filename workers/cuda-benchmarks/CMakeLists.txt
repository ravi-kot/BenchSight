cmake_minimum_required(VERSION 3.24)
project(benchsight_worker LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

find_package(CUDAToolkit REQUIRED)

find_path(CUDNN_INCLUDE_DIR
    NAMES cudnn.h
    HINTS
        "$ENV{CUDNN_ROOT}"
        "$ENV{CUDNN_PATH}"
    PATH_SUFFIXES include
)
find_library(CUDNN_LIBRARY
    NAMES cudnn
    HINTS
        "$ENV{CUDNN_ROOT}"
        "$ENV{CUDNN_PATH}"
    PATH_SUFFIXES lib lib64 lib/x64
)

add_executable(benchsight_worker
    src/main.cpp
    src/vector_add.cu
    src/reduction.cu
    src/gemm.cu
    src/conv2d.cu
    src/softmax.cu
)

target_include_directories(benchsight_worker PRIVATE include)
target_link_libraries(benchsight_worker PRIVATE CUDA::cudart CUDA::cublas)

if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    target_include_directories(benchsight_worker PRIVATE ${CUDNN_INCLUDE_DIR})
    target_link_libraries(benchsight_worker PRIVATE ${CUDNN_LIBRARY})
    target_compile_definitions(benchsight_worker PRIVATE BENCHSIGHT_HAVE_CUDNN=1)
else()
    target_compile_definitions(benchsight_worker PRIVATE BENCHSIGHT_HAVE_CUDNN=0)
endif()

if(TARGET CUDA::nvToolsExt)
    target_link_libraries(benchsight_worker PRIVATE CUDA::nvToolsExt)
endif()
