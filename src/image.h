#ifndef IMAGE_H
#define IMAGE_H

#include "constants.h"

struct image_t {
    const char *file_path;
    char data[WIDTH * HEIGHT];
};

int __host__ read_image(struct image_t *image, const char *file_path);

#endif
