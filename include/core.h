#ifndef CORE_H
#define CORE_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <errno.h>

#define BLACK       "\e[0;30m"
#define RED         "\e[0;31m"
#define GREEN       "\e[0;32m"
#define YELLOW      "\e[0;33m"
#define BLUE        "\e[0;34m"
#define MAGENTA     "\e[0;35m"
#define CYAN        "\e[0;36m"
#define WHITE       "\e[0;37m"
#define COLOR_RESET "\e[0m"

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

// Taken from https://stackoverflow.com/questions/1737726/how-to-perform-rgb-yuv-conversion-in-c-c

#define CLAMP(X, MIN, MAX) ( (X) > MAX ? MAX : (X) < MIN ? MIN : X)

// RGB -> YUV
#define RGB_TO_Y(R, G, B) CLAMP(( (  66 * (R) + 129 * (G) +  25 * (B) + 128) >> 8) +  16, 0, 255)
#define RGB_TO_U(R, G, B) CLAMP(( ( -38 * (R) -  74 * (G) + 112 * (B) + 128) >> 8) + 128, 0, 255)
#define RGB_TO_V(R, G, B) CLAMP(( ( 112 * (R) -  94 * (G) -  18 * (B) + 128) >> 8) + 128, 0, 255)

#endif
