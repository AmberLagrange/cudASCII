#include "ascii.h"

__host__ int init_ascii(struct ascii_t *ascii, int src_width, int scale_width, int src_height, int scale_height, int dark_mode) {

    ascii->scale_width = scale_width;
    ascii->scale_height = scale_height;

    ascii->width = src_width / scale_width;
    ascii->height = src_height / scale_height;

    ascii->dark_mode = dark_mode;

    ascii->data_size = ascii->width * ascii->height;
    ascii->data = (char*)malloc(ascii->data_size);

    ascii->char_set = DEFAULT_CHAR_SET;
    ascii->char_set_size = DEFAULT_CHAR_SET_SIZE;

    return 0;
}

__host__ int cleanup_ascii(struct ascii_t *ascii) {

    free(ascii->data);

    return 0;
}

int set_char_set(struct ascii_t *ascii, const char* char_set, size_t char_set_size) {
    
    ascii->char_set      = char_set;
    ascii->char_set_size = char_set_size;

    return 0;
}
