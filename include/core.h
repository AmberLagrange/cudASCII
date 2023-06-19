#ifndef CORE_H
#define CORE_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <errno.h>

#define DEFAULT_CHAR_SET      "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`\'. "
#define DEFAULT_CHAR_SET_SIZE sizeof(DEFAULT_CHAR_SET) - 1  // Won't move over the null terminator. This is fine as the
                                                            // DEFAULT_CHAR_SET is an array of characters and shouldn't
                                                            // be used as a string

#define SCALE_WIDTH  4
#define SCALE_HEIGHT 4

#ifndef E_OK
#define E_OK 0
#endif

#ifndef E_FILE
#define E_FILE -1
#endif

#ifndef E_TEST
#define E_TEST -1
#endif

#define gpu_check_error(code) { gpu_assert((code), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line);

#endif