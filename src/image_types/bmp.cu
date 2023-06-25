#include "bmp.h"

__host__ int load_bmp(bmp_t *bmp, const char *filepath) {

    FILE *file = fopen(filepath, "rb");

    if (!file) {
        fprintf(stderr, "Could not read from file %s: %s\n", filepath, strerror(errno));
        return E_FILE_READ;
    }

    size_t data_read = fread(&(bmp->header), 1, sizeof(bmp_header_t), file);
    if (data_read != sizeof(bmp_header_t)) {
        fprintf(stderr,
                "Did not read header from %s properly\n"
                "Expected %lu bytes. Read %lu bytes\n",
                filepath, sizeof(bmp_header_t), data_read);
        return E_FILE_READ;
    }

    size_t data_size = bmp->header.file_size - sizeof(bmp_header_t);
    bmp->pixels = (u8*)malloc(data_size);
    data_read = fread(bmp->pixels, 1, data_size, file);
    if (data_read != data_size) {
        fprintf(stderr, "Did not read all data from %s\n", filepath);
        return E_FILE_READ;
    }

    fclose(file);
    return data_size;
}

__host__ void print_bmp_header(bmp_header_t *header) {

    printf(
            "Signature:   %c%c\n"
            "File Size:   %d Bytes\n"
            "Reserved:    %d\n"
            "Data Offset: %d\n"
            "\n"
            "Info Header Size: %d\n"
            "\n"
            "Width:  %d\n"
            "Height: %d\n"
            "Planes: %d\n"
            "BPP:    %d\n"
            "\n"
            "Compression: %s\n"
            "Image Size:  %d\n"
            "\n"
            "Horizontal Pixels Per Meter: %d\n"
            "Vertical Pixels Per Meter:   %d\n"
            "\n"
            "Colors Used:      %d\n"
            "Important Colors: %d\n",
            header->signature & 0xFF, header->signature >> CHAR_BIT,
            header->file_size,
            header->reserved,
            header->data_offset,

            header->info_header_size,

            header->width,
            header->height,
            header->planes,
            header->bpp,

            (header->compression == 0) ? "No Compression" : ((header->compression == 1) ? "BI_RLE8 8bit RLE" : "BI_RLE4 4bit RLE"),
            header->image_size,

            header->x_ppm,
            header->y_ppm,

            header->colors_used,
            header->important_colors
            );
}
