#include "image.h"

#include <stdio.h>

int __host__ read_image(struct image_t *image, const char *file_path) {
    image->file_path = file_path;

    // Temp for "reading in" data of image
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        image->data[i] = 0;
    }

    printf("Loaded in image at filepath %s\n", image->file_path);

    return 0;
}
