#include "image.h"

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// TODO: Abstract away from bmp only

int __host__ read_image(struct image_t *image, const char *filepath) {

    image->filepath = filepath;

    bmp_t bmp;
    size_t data_size = load_bmp(&bmp, filepath);
    //print_bmp_header(&(bmp.header));

    image->width = bmp.header.width;
    image->height = bmp.header.height;
    image->bytes_per_pixel = bmp.header.bpp / (CHAR_BIT);
    image->data_size = data_size;

    image->data = bmp.pixels;

    return 0;
}

int __host__ cleanup_image(struct image_t *image) {

    free(image->data);

    return 0;
}
