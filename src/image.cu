#include "image.h"

// TODO: Abstract away from bmp only

int __host__ read_image(image_t *image, const char *filepath) {

    image->filepath = filepath;

    return read_bmp_image(image, filepath);
}

int __host__ cleanup_image(image_t *image) {

    free(image->data);

    return E_OK;
}

int __host__ read_bmp_image(image_t *image, const char *filepath) {

    bmp_t bmp;
    
    int data_size = load_bmp(&bmp, filepath);

    if (data_size <= 0) {
        return data_size;
    }
    
    print_bmp_header(&(bmp.header));

    image->width = bmp.header.width;
    image->height = bmp.header.height;

    image->bytes_per_pixel = bmp.header.bpp / (CHAR_BIT);
    image->color_format = COLOR_RGB;

    image->data_size = data_size;

    image->data = bmp.pixels;

    return E_OK;
}

int __host__ read_png_image(image_t *image, const char *filepath) {

    fprintf(stderr, "Png image not supported\n");
    return E_FILE_READ;
}
