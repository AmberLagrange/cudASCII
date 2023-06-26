#include "image_types.h"

#include "../image.h"

const image_type_t image_types[] = {
    image_type_t {
        "Bitmap",
        {
            ".bmp"
        },
        {
            "\x42\x4D"
        },
        &read_bmp_image
    },

    image_type_t {
        "PNG",
        {
            ".png"
        },
        {
            "\x89\x50\x4E\x47\x00D\x0A\x1A\x0A"
        },
        &read_png_image
    },

    image_type_t {
        "JPEG",
        {
            ".jpg",
            ".jpeg"
        },
        {
            "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46\x00\x01"
        },
        &read_jpg_image
    },
};

const size_t num_image_types = sizeof(image_types) / sizeof(image_type_t);