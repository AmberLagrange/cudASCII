#include "bmp.h"

__host__ int load_bmp(struct bmp_t *bmp, const char *filepath) {

    FILE *file = fopen(filepath, "rb");
    fread(&(bmp->header), sizeof(bmp_header_t), 1, file);

    bmp->pixels = (uint8_t*)malloc(bmp->header.width * bmp->header.height * (bmp->header.bpp / CHAR_BIT));

    fread(bmp->pixels, bmp->header.width * bmp->header.height * (bmp->header.bpp / CHAR_BIT), 1, file);

    fclose(file);

    return 0;
}

__host__ void print_bmp_header(struct bmp_header_t *header) {

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