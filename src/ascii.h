#ifndef ASCII_H
#define ASCII_H

#include "core.h"

#include <stdint.h>

struct ascii_t {
    int scale_width;
    int scale_height;

    int width;
    int height;

    int dark_mode;

    const char *char_set;
    size_t     char_set_size;

    char  *data;
    size_t data_size;
};

__host__ int init_ascii(struct ascii_t *ascii, int src_width, int scale_width, int src_height, int scale_height, int dark_mode);
__host__ int cleanup_ascii(struct ascii_t *ascii);

__host__ int set_char_set(struct ascii_t *ascii, const char* char_set);

#endif
