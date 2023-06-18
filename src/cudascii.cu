#include <stdio.h>

#include "core.h"

#include "image.h"
#include "ascii.h"

inline void gpu_assert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

__global__ void convert_to_ascii(struct ascii_t *ascii, struct image_t *image) {

    int average = 0;

    int row = threadIdx.x;
    int col = blockIdx.x;

    for (int r = row * ascii->scale_height; r < (row + 1) * ascii->scale_height; ++r) {
        for (int c = col * ascii->scale_width; c < (col + 1) * ascii->scale_width; ++c) {
            for (int i = 0; i < image->bytes_per_pixel; ++i) {
                average += image->data[(r * image->width + c) * image->bytes_per_pixel + i];
            }
        }
    }

    average /= (image->bytes_per_pixel * ascii->scale_width * ascii->scale_height);
    average = (ascii->dark_mode) ? (255 - average) : average;

    ascii->data[row * ascii->width + col] = ascii->char_set[(average * (ascii->char_set_size - 1)) / 255];

    return;
}

__host__ int test_blank(struct ascii_t *ascii, const char *greyscale) {

    char blank_character = ascii->dark_mode ? greyscale[0] : greyscale[strlen(greyscale) - 1];

    for (int i = 0; i < ascii->data_size; ++i) {
        if (ascii->data[i] != greyscale[0]) {
            printf("Error! Did not copy over properly!\nIndex %d has value %d\n", i, ascii->data[i]);
            return -1;
        }
    }

    printf("All values were copied over successfully!\n");

    return 0;
}

__host__ int write_image(struct ascii_t *ascii, const char *filepath) {

    FILE *file = fopen(filepath, "w+");

    for (int row = ascii->height - 1; row >= 0 ; --row) {
        for (int col = 0; col < ascii->width; ++col) {
            fprintf(file, "%c", ascii->data[row * ascii->width + col]);
        }
        fprintf(file, "\n");
    }

    return 0;
}

__host__ int image_to_ascii(ascii_t *h_ascii, const char *filepath) {

    // Host image data
    struct image_t h_image;
    read_image(&h_image, filepath);

    // Host ascii data
    init_ascii(h_ascii, h_image.width, SCALE_WIDTH, h_image.height, SCALE_HEIGHT, 1);

    // Pointers to data on device
    struct image_t *d_image;
    struct ascii_t *d_ascii;

    // Create and copy image struct over
    gpu_check_error(cudaMallocHost(&d_image, sizeof(struct image_t)));
    gpu_check_error(cudaMemcpy(d_image, &h_image, sizeof(h_image), cudaMemcpyHostToDevice));

    // Create and copy image data over
    gpu_check_error(cudaMallocHost(&(d_image->data), d_image->data_size));
    gpu_check_error(cudaMemcpy(d_image->data, h_image.data, d_image->data_size, cudaMemcpyHostToDevice));

    // Create and copy ascii struct data
    gpu_check_error(cudaMallocHost(&d_ascii, sizeof(struct ascii_t)));
    gpu_check_error(cudaMemcpy(d_ascii, h_ascii, sizeof(struct ascii_t), cudaMemcpyHostToDevice));

    // Create and copy ascii char set data
    gpu_check_error(cudaMallocHost(&(d_ascii->char_set), d_ascii->char_set_size));
    gpu_check_error(cudaMemcpy((char*)d_ascii->char_set, h_ascii->char_set, d_ascii->char_set_size, cudaMemcpyHostToDevice));

    // Create ascii data
    gpu_check_error(cudaMallocHost(&(d_ascii->data), d_ascii->data_size));

    // Run the kernel
    convert_to_ascii<<<d_ascii->width, d_ascii->height>>>(d_ascii, d_image);

    // Copy ascii data from device to host
    gpu_check_error(cudaMemcpy(h_ascii->data, d_ascii->data, h_ascii->data_size, cudaMemcpyDeviceToHost));
    
    // Clean up cuda memory
    gpu_check_error(cudaFreeHost(d_ascii->data));
    gpu_check_error(cudaFreeHost((char*)d_ascii->char_set));
    gpu_check_error(cudaFreeHost(d_ascii));
    gpu_check_error(cudaFreeHost(d_image->data));
    gpu_check_error(cudaFreeHost(d_image));

    cudaDeviceReset();

    // Clean up host memory
    cleanup_image(&h_image);

    return 0;
}

int main(int argc, char **argv) {

    if (argc < 2) {
        fprintf(stderr, "Did not provide an image filepath!\n");
        return -1;
    }

    if (argc < 3) {
        fprintf(stderr, "Did not provide a filepath to write to!\n");
        return -2;
    }

    struct ascii_t ascii;

    image_to_ascii(&ascii, argv[1]);
    write_image(&ascii, argv[2]);
    cleanup_ascii(&ascii);

    return 0;
}
