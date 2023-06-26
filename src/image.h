#ifndef IMAGE_H
#define IMAGE_H

#include "core.h"

#include <stdio.h>

#include "image_types/bmp.h"
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

int __host__ read_image(image_t *image, const char *filepath);
int __host__ cleanup_image(image_t *image);

int __host__ read_bmp_image(image_t *image, FILE *file);
int __host__ read_png_image(image_t *image, FILE *file);
int __host__ read_jpg_image(image_t *image, FILE *file);

#endif
