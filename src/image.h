#ifndef IMAGE_H
#define IMAGE_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "constants.h"
#include <stdint.h>

struct image_t {
    const char *file_path;

    int width;
    int height;

    uint8_t *data;
};

int __host__ read_image(struct image_t *image, const char *file_path);
int __host__ cleanup_image(struct image_t *image);

#endif
