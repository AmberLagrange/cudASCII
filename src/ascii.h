#ifndef ASCII_H
#define ASCII_H

#include "cuda_runtime.h"

#include <stdint.h>
#include "constants.h"

struct ascii_t {
    int scale_width;
    int scale_height;

    int width;
    int height;

    char *data;

    int dark_mode;
};

__host__ int init_ascii(struct ascii_t *ascii, int src_width, int scale_width, int src_height, int scale_height, int dark_mode);
__host__ int cleanup_ascii(struct ascii_t *ascii);

#endif
