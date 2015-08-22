project(Earthquake_Forecaster)
cmake_minimum_required(VERSION 3.2.1)
find_package(CUDA REQUIRED)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/headers)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/kernels)
LINK_DIRECTORIES(/usr/local/cuda/lib64)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})


set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}; " -O3 -ffast-math -Wall")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};  -std=c++11 --compiler-options -g -lineinfo -use_fast_math -Xptxas "-dlcm=cg" -maxrregcount=32)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_30,code=sm_30)

FILE(GLOB SOURCES "src/*")
FILE(GLOB HEADERS  "headers/*")
FILE(GLOB KERNELS "kernels/*")

CUDA_ADD_LIBRARY(NN ${KERNELS} src/network.cu)
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} NN)
