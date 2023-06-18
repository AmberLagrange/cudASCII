#ifndef CORE_H
#define CORE_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DEFAULT_CHAR_SET      "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`\'. "
#define DEFAULT_CHAR_SET_SIZE sizeof(DEFAULT_CHAR_SET) - 1  // Won't move over the null terminator. This is fine as the
                                                            // DEFAULT_CHAR_SET is an array of characters and shouldn't
                                                            // be used as a string

#define SCALE_WIDTH  4
#define SCALE_HEIGHT 4

#define gpu_check_error(code) { gpu_assert((code), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line);

#endif
