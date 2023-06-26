#ifndef IMAGE_H
#define IMAGE_H

#include "core.h"

#include <stdio.h>

#include "color_format.h"

typedef struct image_t {
    const char *filepath;

    int width;
    int height;
    
    int bytes_per_pixel;
    color_format_t color_format;

    u8    *data;
    size_t data_size;
} image_t;

__host__ __device__ void rgb_to_yuv(u8 *byte_1, u8 *byte_2, u8 *byte_3);
__host__ __device__ void yuv_to_rgb(u8 *byte_1, u8 *byte_2, u8 *byte_3);
__host__ __device__ void rgb_to_rgb(u8 *byte_1, u8 *byte_2, u8 *byte_3);
__host__ __device__ void yuv_to_yuv(u8 *byte_1, u8 *byte_2, u8 *byte_3);

__host__ int read_image(image_t *image, const char *filepath);
__host__ int cleanup_image(image_t *image);

__host__ int read_bmp_image(image_t *image, FILE *file);
__host__ int read_png_image(image_t *image, FILE *file);
__host__ int read_jpg_image(image_t *image, FILE *file);

#endif
