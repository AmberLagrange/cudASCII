#include "ascii.h"

#include <stdio.h>

__host__ int init_ascii(struct ascii_t *ascii, int src_width, int scale_width, int src_height, int scale_height) {

    ascii->width = src_width / scale_width;
    ascii->height = src_height / scale_height;

    ascii->data = (char*)malloc(ascii->width * ascii->height);

    printf("Scaled image size is: %dx%d\n", ascii->width, ascii->height);

    return 0;
}

__host__ int cleanup_ascii(struct ascii_t *ascii) {

    free(ascii->data);

    return 0;
}
