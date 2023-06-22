#ifndef BMP_H
#define BMP_H

#include "core.h"

#include <stdio.h>

typedef struct bmp_header_t {
    // Header
    u16 signature;
    u32 file_size;
    u32 reserved;
    u32 data_offset;

    // Info Header
    u32 info_header_size;
    u32 width;
    u32 height;
    u16 planes;
    u16 bpp;
    u32 compression;
    u32 image_size;
    u32 x_ppm;
    u32 y_ppm;
    u32 colors_used;
    u32 important_colors;
} __attribute__((packed)) bmp_header_t;

typedef struct bmp_t {
    bmp_header_t header;
    u8 *pixels;
} bmp_t;

__host__ int load_bmp(bmp_t *bmp, const char *filepath);
__host__ int cleanup_bmp(bmp_t *bmp);

void __host__ print_bmp_header(bmp_header_t *bmp);

#endif
