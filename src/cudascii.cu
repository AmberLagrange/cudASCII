#include <stdio.h>

#include "core.h"

#include "ascii.h"
#include "renderer/cuda_renderer.h"

__host__ int test_blank(ascii_t *ascii, const char *greyscale) {

    char blank_character = ascii->dark_mode ? greyscale[0] : greyscale[strlen(greyscale) - 1];

    for (int i = 0; i < ascii->data_size; ++i) {
        if (ascii->data[i] != greyscale[0]) {
            printf("Error! Did not copy over properly!\nIndex %d has value %d\n", i, ascii->data[i]);
            return E_TEST_FAILED;
        }
    }

    printf("All values were copied over successfully!\n");

    return E_OK;
}

__host__ int write_image(ascii_t *ascii, const char *filepath) {

    FILE *file;

    if (!filepath) {
        file = stdout;
    } else {
        file = fopen(filepath, "w+");

        if (!file) {
            fprintf(stderr, "Could not write image to file %s: %s\n", filepath, strerror(errno));
            return E_FILE_WRITE;
        }
    }

    for (int row = ascii->height - 1; row >= 0 ; --row) {
        for (int col = 0; col < ascii->width; ++col) {
            fprintf(file, "%c", ascii->data[row * ascii->width + col]);
        }
        fprintf(file, "\n");
    }

    fclose(file);

    return E_OK;
}

__host__ int write_color_image(ascii_t *ascii, const char *filepath) {

    FILE *file;

    if (!filepath) {
        file = stdout;
    } else {
        file = fopen(filepath, "w+");

        if (!file) {
            fprintf(stderr, "Could not write image to file %s: %s\n", filepath, strerror(errno));
            return E_FILE_WRITE;
        }
    }

    if (!ascii->color_enabled) {
        fprintf(stderr, "Color not enabled\n");
        return E_COLOR_ENABLED;
    }

    for (int row = ascii->height - 1; row >= 0 ; --row) {
        for (int col = 0; col < ascii->width; ++col) {

            u8 r = ascii->r_data[row * ascii->width + col];
            u8 g = ascii->g_data[row * ascii->width + col];
            u8 b = ascii->b_data[row * ascii->width + col];

            char c = ascii->data[row * ascii->width + col];

            fprintf(file, "\033[38;2;%d;%d;%dm%c\x1b[0m", r, g, b, c);
        }
        fprintf(file, "\n");
    }

    fclose(file);

    return E_OK;
}

__host__ int main(int argc, char **argv) {

    if (argc < 2) {
        fprintf(stderr, "Did not provide an image filepath!\n");
        return E_INVALID_PARAMS;
    }

    const char *filepath = NULL;

    if (argc < 3) {
        printf("Did not provide a filepath to write to. Using stdout\n");
    } else {
        filepath = argv[2];
    }

    ascii_t ascii;
    int ret;

    init_ascii(&ascii, 1920 / 4, 2560 / 4, 0);
    enable_color(&ascii);

    if (ret = image_to_ascii(&ascii, argv[1])) {
        return ret;
    }

    if (ascii.color_enabled) {
        if (ret = write_color_image(&ascii, filepath)) {
            return ret;
        }
    } else {
        if (ret = write_image(&ascii, filepath)) {
            return ret;
        }
    }

    cleanup_ascii(&ascii);

    return E_OK;
}
