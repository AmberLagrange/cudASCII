#include "image.h"

#include <string.h>

#include "image_types/image_types.h"
#include "image_types/bmp.h"

// Color Format Conversions

__host__ __device__ void rgb_to_yuv(u8 *byte_1, u8 *byte_2, u8 *byte_3) {

    u8 r = *byte_1;
    u8 g = *byte_2;
    u8 b = *byte_3;

    *byte_1 = RGB_TO_Y(r, g, b);
    *byte_2 = RGB_TO_U(r, g, b);
    *byte_3 = RGB_TO_V(r, g, b);
}

__host__ __device__ void yuv_to_rgb(u8 *byte_1, u8 *byte_2, u8 *byte_3) {

    u8 y = *byte_1;
    u8 u = *byte_2;
    u8 v = *byte_3;

    *byte_1 = YUV_TO_R(y, u, v);
    *byte_2 = YUV_TO_G(y, u, v);
    *byte_3 = YUV_TO_B(y, u, v);
}

__host__ __device__ void rgb_to_rgb(u8 *byte_1, u8 *byte_2, u8 *byte_3) {
    // NOP
}

__host__ __device__ void yuv_to_yuv(u8 *byte_1, u8 *byte_2, u8 *byte_3) {
    // NOP
}

// TODO: Add more image types

__host__ int read_image(image_t *image, const char *filepath) {

    const char *file_extension = strrchr(filepath, '.');

    if (!file_extension) {
        fprintf(stderr, "No file extension found.\n");
        return E_INVALID_EXTENSION;
    }

    int (*call_back)(image_t*, FILE*) = NULL;

    image_type_t image_type;
    size_t bytes_read;
    size_t signature_size;
    char signature[MAX_SIGNATURE_SIZE];

    FILE *file = fopen(filepath, "rb");
    if (!file) {
        fprintf(stderr, "Could not read from file %s: %s\n", filepath, strerror(errno));
        return E_FILE_READ;
    }

    int found;
    for (size_t i = 0; i < num_image_types; ++i) {

        image_type = image_types[i];

        found = 0;
        for (size_t i = 0; i < MAX_FILE_EXTENSIONS; ++i) {

            if (!image_type.file_extensions[i]) {
                break;
            }

            if (strcmp(image_type.file_extensions[i], file_extension) != 0) {
                continue;
            }
            found = 1;
            break;
        }


        if (!found) {
            continue;
        }

        found = 0;
        for (size_t i = 0; i < MAX_SIGNATURES; ++i) {

            if (!image_type.signatures[i]) {
                break;
            }

            memset(signature, 0, sizeof(signature));
            signature_size = strlen(image_type.signatures[i]);
            bytes_read = fread(signature, 1, signature_size, file);
            rewind(file);

            if (bytes_read != signature_size) {
                fprintf(stderr, "Did not read entire signature from %s.\n", filepath);
                fclose(file);
                return E_FILE_READ;
            }

            if (strcmp(image_type.signatures[i], signature) != 0) {
                continue;
            }

            call_back = image_type.call_back;
            found = 1;
            break;
        }

        if (found) {
            break;
        }
    }

    if (!call_back) {
        fprintf(stderr, "Could not read the file extension %s\n", file_extension);
        fclose(file);
        return E_INVALID_EXTENSION;
    }

    image->filepath = filepath;

    int ret = call_back(image, file);
    fclose(file);

    return ret;
}

__host__ int cleanup_image(image_t *image) {

    free(image->data);

    return E_OK;
}

__host__ int read_bmp_image(image_t *image, FILE *file) {

    bmp_t bmp;
    bmp.filepath = image->filepath;
    
    int data_size = load_bmp(&bmp, file);

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

__host__ int read_png_image(image_t *image, FILE *file) {

    fprintf(stderr, "PNG image format not supported\n");
    return E_FILE_READ;
}

__host__ int read_jpg_image(image_t *image, FILE *file) {

    fprintf(stderr, "JPEG image format not supported\n");
    return E_FILE_READ;
}
