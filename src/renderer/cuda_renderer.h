#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H

#include "core.h"

#include "../ascii.h"
#include "../image/image.h"

#define MAX_THREADS_X_DIM     1024
#define MAX_THREADS_Y_DIM     1024
#define MAX_THREADS_Z_DIM       64
#define MAX_THREADS_PER_BLOCK 1024

__global__ void convert_to_ascii(ascii_t *ascii, image_t *image, size_t index, volatile int *error);
__host__ int image_to_ascii(ascii_t *h_ascii, const char *filepath);

#endif
