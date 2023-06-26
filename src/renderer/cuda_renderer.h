#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H

#include "core.h"

#include "../ascii.h"
#include "../image/image.h"

__global__ void convert_to_ascii(ascii_t *ascii, image_t *image, volatile int *error);
__host__ int image_to_ascii(ascii_t *h_ascii, const char *filepath);

#endif
