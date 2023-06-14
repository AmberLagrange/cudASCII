cmake_minimum_required(VERSION 3.24)

set (CMAKE_C_STANDARD 11)
project(cudASCII LANGUAGES C CUDA)

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

add_executable("${PROJECT_NAME}"
    src/cudASCII.cu)

set_target_properties(cudASCII PROPERTIES CUDA_ARCHITECTURES "native")