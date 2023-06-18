#ifndef BMP_H
#define BMP_H

#include "cuda_runtime.h"

#include <stdint.h>
#include <stdio.h>

struct bmp_header_t {
    // Header
    uint16_t signature;
    uint32_t file_size;
    uint32_t reserved;
    uint32_t data_offset;

    // Info Header
    uint32_t info_header_size;
    uint32_t width;
    uint32_t height;
    uint16_t planes;
    uint16_t bpp;
    uint32_t compression;
    uint32_t image_size;
    uint32_t x_ppm;
    uint32_t y_ppm;
    uint32_t colors_used;
    uint32_t important_colors;
} __attribute__((packed));

struct bmp_t {
    struct bmp_header_t header;
    uint8_t *pixels;
};

__host__ int load_bmp(struct bmp_t *bmp, const char *filepath);
__host__ int cleanup_bmp(struct bmp_t *bmp);

void __host__ print_bmp_header(struct bmp_header_t *bmp);

#endif