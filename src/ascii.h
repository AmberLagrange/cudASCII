#ifndef ASCII_H
#define ASCII_H

#include "core.h"

#include "image/color_format.h"

typedef struct ascii_t {
    size_t scale_width;
    size_t scale_height;

    size_t width;
    size_t height;

    int dark_mode;
    int color_enabled;

    char  *char_set;
    size_t char_set_size;

    char *data; /// ASCII Character

    u8 *y_data; // Luminance
    u8 *u_data; // Blue projection
    u8 *v_data; // Red projection
    size_t data_size;
} ascii_t;

__host__ int init_ascii(ascii_t *ascii, size_t width, size_t height, int dark_mode);
__host__ int cleanup_ascii(ascii_t *ascii);

__host__ int set_char_set(ascii_t *ascii, const char* char_set);
__host__ int enable_color(ascii_t *ascii);

#endif
