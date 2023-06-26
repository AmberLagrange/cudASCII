cmake_minimum_required(VERSION 3.24)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
project(cudASCII LANGUAGES C CUDA)

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

add_executable("${PROJECT_NAME}"
    src/cudascii.cu
    src/ascii.cu
    src/image/image.cu 
    src/image/image_types/image_types.cu
    src/image/image_types/bmp.cu
    src/renderer/cuda_renderer.cu
    )

target_precompile_headers(cudASCII PRIVATE include/core.hpp)

set_target_properties(cudASCII PROPERTIES CUDA_ARCHITECTURES "native")
