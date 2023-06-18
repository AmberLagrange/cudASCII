#include "image.h"

#include <stdio.h>

int __host__ read_image(struct image_t *image, const char *file_path) {
    image->file_path = file_path;

    // Temp for reading header for width/height
    image->width = SRC_WIDTH;
    image->height = SRC_HEIGHT;

    image->data = (uint8_t*)malloc(image->width * image->height);

    // Temp for "reading in" data of image
    for (int i = 0; i < SRC_WIDTH * SRC_HEIGHT; ++i) {
        image->data[i] = 0;
    }

    printf("Loaded in image at filepath %s\n"
           "Image source size is: %dx%d\n",
           image->file_path, image->width, image->height);

    return 0;
}

int __host__ cleanup_image(struct image_t *image) {
    free(image->data);

    return 0;
}
