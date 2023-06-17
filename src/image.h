#ifndef IMAGE_H
#define IMAGE_H

#include "constants.h"
#include <stdint.h>

struct image_t {
    const char *file_path;
    uint8_t *data;
    
    int src_width;
    int src_height;
    
    int scale_width;
    int scale_height;
};

int __host__ read_image(struct image_t *image, const char *file_path);
int __host__ cleanup_image(struct image_t *image);

#endif
