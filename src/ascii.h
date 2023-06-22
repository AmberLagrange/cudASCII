#ifndef ASCII_H
#define ASCII_H

#include "core.h"

#include "color_format.h"

typedef struct ascii_t {
    int scale_width;
    int scale_height;

    int width;
    int height;

    int dark_mode;
    int color_enabled;

    const char *char_set;
    size_t      char_set_size;

    char  *y_data; // Luminance
    char  *u_data; // Blue projection
    char  *v_data; // Red projection
    size_t data_size;
} ascii_t;

__host__ int init_ascii(ascii_t *ascii, int src_width, int scale_width, int src_height, int scale_height, int dark_mode);
__host__ int cleanup_ascii(ascii_t *ascii);

__host__ int set_char_set(ascii_t *ascii, const char* char_set);
__host__ int enable_color(ascii_t *ascii);

#endif
