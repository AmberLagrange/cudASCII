#include "image.h"

#include <string.h>

#include "image_types/image_types.h"

// TODO: Add more images

int __host__ read_image(image_t *image, const char *filepath) {

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

            signature_size = strlen(image_type.signatures[i]);
            bytes_read = fread(signature, 1, signature_size, file);
            rewind(file);

            if (bytes_read != signature_size) {
                fprintf(stderr, "Did not read entire signature from %s.\n", filepath);
                fclose(file);
                return E_FILE_READ;
            }

            if (strcmp(image_type.signatures[0], signature) != 0) {
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

int __host__ cleanup_image(image_t *image) {

    free(image->data);

    return E_OK;
}

int __host__ read_bmp_image(image_t *image, FILE *file) {

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

int __host__ read_png_image(image_t *image, FILE *file) {

    fprintf(stderr, "PNG image format not supported\n");
    return E_FILE_READ;
}

int __host__ read_jpg_image(image_t *image, FILE *file) {

    fprintf(stderr, "JPG image format not supported\n");
    return E_FILE_READ;
}
