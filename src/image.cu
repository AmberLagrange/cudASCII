#include "image.h"

#include <stdio.h>

int __host__ read_image(struct image_t *image, const char *file_path) {
    image->file_path = file_path;

    // Temp for reading header for width/height
    image->src_width = SRC_WIDTH;
    image->src_height = SRC_HEIGHT;

    image->scale_width = image->src_width / SCALE_WIDTH;
    image->scale_height = image->src_height / SCALE_HEIGHT;

    image->data = (uint8_t*)malloc(image->src_width * image->src_height);

    // Temp for "reading in" data of image
    for (int i = 0; i < SRC_WIDTH * SRC_HEIGHT; ++i) {
        image->data[i] = 0;
    }

    printf("Loaded in image at filepath %s\n"
           "Image source size is: %dx%d\n"
           "Scaled image size is: %dx%d\n",
           image->file_path, image->src_width, image->src_height, image->scale_width, image->scale_height);

    return 0;
}

int __host__ cleanup_image(struct image_t *image) {
    free(image->data);

    return 0;
}
