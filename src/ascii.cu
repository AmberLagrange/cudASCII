#include "ascii.h"

__host__ int init_ascii(ascii_t *ascii, int src_width, int scale_width, int src_height, int scale_height, int dark_mode) {

    ascii->scale_width = scale_width;
    ascii->scale_height = scale_height;

    ascii->width = src_width / scale_width;
    ascii->height = src_height / scale_height;

    ascii->dark_mode = dark_mode;
    ascii->color_enabled = 0;

    ascii->data_size = ascii->width * ascii->height;
    ascii->y_data = (char*)malloc(ascii->data_size);
    ascii->u_data = NULL;
    ascii->v_data = NULL;

    ascii->char_set_size = DEFAULT_CHAR_SET_SIZE;
    ascii->char_set = (char*)malloc(ascii->char_set_size);
    memcpy(ascii->char_set, DEFAULT_CHAR_SET, ascii->char_set_size);

    return E_OK;
}

__host__ int cleanup_ascii(ascii_t *ascii) {

    free(ascii->char_set);

    free(ascii->y_data);
    free(ascii->u_data);
    free(ascii->v_data);

    return E_OK;
}

int set_char_set(ascii_t *ascii, const char *char_set, size_t char_set_size) {
    
    free(ascii->char_set);
    ascii->char_set_size = char_set_size;
    ascii->char_set = (char*)malloc(ascii->char_set_size);
    memcpy(ascii->char_set, char_set, ascii->char_set_size);

    return E_OK;
}

int enable_color(ascii_t *ascii) {

    if (ascii->color_enabled) {
        return E_COLOR_ENABLED;
    }

    ascii->color_enabled = 1;

    ascii->u_data = (char*)malloc(ascii->data_size);
    ascii->v_data = (char*)malloc(ascii->data_size);

    return E_OK;
}
