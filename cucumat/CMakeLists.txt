cmake_minimum_required(VERSION 3.9)
project(cucumat)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
find_package(CUDA REQUIRED)
# set(CUDA_NVCC_FLAGS -arch=sm_75 -O3)

set(SRC
    ${PROJECT_SOURCE_DIR}/cuda/cucumat.cu
    ${PROJECT_SOURCE_DIR}/cuda/cucumat_kernel.cu)
set(INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/cuda)

cuda_add_library(cucumat SHARED ${INCLUDE_DIRS} ${SRC})
target_link_libraries(cucumat cublas)
