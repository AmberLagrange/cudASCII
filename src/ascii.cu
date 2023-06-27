#include "ascii.h"

__host__ int init_ascii(ascii_t *ascii, size_t width, size_t height, int dark_mode) {

    ascii->scale_width = 0;
    ascii->scale_height = 0;

    ascii->width = width;
    ascii->height = height;

    ascii->dark_mode = dark_mode;
    ascii->color_enabled = 0;

    ascii->data_size = ascii->width * ascii->height;
    ascii->data = (char*)malloc(ascii->data_size);

    ascii->y_data = NULL;
    ascii->u_data = NULL;
    ascii->v_data = NULL;

    ascii->char_set_size = DEFAULT_CHAR_SET_SIZE;
    ascii->char_set = (char*)malloc(ascii->char_set_size);
    memcpy(ascii->char_set, DEFAULT_CHAR_SET, ascii->char_set_size);

    return E_OK;
}

__host__ int cleanup_ascii(ascii_t *ascii) {

    free(ascii->char_set);

    free(ascii->data);
    
    free(ascii->y_data);
    free(ascii->u_data);
    free(ascii->v_data);

    return E_OK;
}

__host__ int set_char_set(ascii_t *ascii, const char *char_set, size_t char_set_size) {
    
    free(ascii->char_set);
    ascii->char_set_size = char_set_size;
    ascii->char_set = (char*)malloc(ascii->char_set_size);
    memcpy(ascii->char_set, char_set, ascii->char_set_size);

    return E_OK;
}

__host__ int enable_color(ascii_t *ascii) {

    if (ascii->color_enabled) {
        return E_COLOR_ENABLED;
    }

    ascii->color_enabled = 1;

    ascii->y_data = (u8*)malloc(ascii->data_size);
    ascii->u_data = (u8*)malloc(ascii->data_size);
    ascii->v_data = (u8*)malloc(ascii->data_size);

    return E_OK;
}
