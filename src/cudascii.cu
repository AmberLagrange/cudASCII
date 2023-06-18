#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "constants.h"

#include "image.h"
#include "ascii.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void convert_to_ascii(char *greyscale, struct image_t *image, struct ascii_t *ascii) {

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

    ascii->data[row * ascii->width + col] = greyscale[(average * (NUM_ASCII - 2)) / 255];

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

__host__ int test_shrek(struct ascii_t *ascii) {

    for (int row = ascii->height - 1; row >= 0 ; --row) {
        for (int col = 0; col < ascii->width; ++col) {
            printf("%c", ascii->data[row * ascii->width + col]);
        }
        printf("\n");
    }

    return 0;
}

__host__ int image_to_ascii(ascii_t *h_ascii, const char *filepath, const char *h_greyscale) {

    // Host image data
    struct image_t h_image;
    read_image(&h_image, filepath);

    // Host ascii data
    init_ascii(h_ascii, h_image.width, SCALE_WIDTH, h_image.height, SCALE_HEIGHT, 1);

    // Pointers to data on device
    char       *d_greyscale;
    struct image_t *d_image;
    struct ascii_t *d_ascii;

    // Create and copy greyscale data over
    gpuErrchk(cudaMallocHost(&d_greyscale, NUM_ASCII));
    gpuErrchk(cudaMemcpy(d_greyscale, h_greyscale , NUM_ASCII, cudaMemcpyHostToDevice));

    // Create and copy image struct over
    gpuErrchk(cudaMallocHost(&d_image, sizeof(struct image_t)));
    gpuErrchk(cudaMemcpy(d_image, &h_image, sizeof(h_image), cudaMemcpyHostToDevice));

    // Create and copy image data over
    gpuErrchk(cudaMallocHost(&(d_image->data), d_image->data_size));
    gpuErrchk(cudaMemcpy(d_image->data, h_image.data, d_image->data_size, cudaMemcpyHostToDevice));

    // Create and copy ascii struct data
    gpuErrchk(cudaMallocHost(&d_ascii, sizeof(struct ascii_t)));
    gpuErrchk(cudaMemcpy(d_ascii, h_ascii, sizeof(struct ascii_t), cudaMemcpyHostToDevice));

    // Create ascii data
    gpuErrchk(cudaMallocHost(&(d_ascii->data), d_ascii->data_size));

    // Run the kernel
    convert_to_ascii<<<d_ascii->width, d_ascii->height>>>(d_greyscale, d_image, d_ascii);

    // Copy ascii data from device to host
    gpuErrchk(cudaMemcpy(h_ascii->data, d_ascii->data, h_ascii->data_size, cudaMemcpyDeviceToHost));
    
    // Clean up cuda memory
    gpuErrchk(cudaFreeHost(d_ascii->data));
    gpuErrchk(cudaFreeHost(d_ascii));
    gpuErrchk(cudaFreeHost(d_image->data));
    gpuErrchk(cudaFreeHost(d_image));
    gpuErrchk(cudaFreeHost(d_greyscale));

    cudaDeviceReset();

    // Clean up host memory
    cleanup_image(&h_image);

    return 0;
}

int main(int argc, char **argv) {

    if (argc < 2) {
        printf("Did not provide a filepath\n");
        return -1;
    }

    // Host greyscale data
    const char *greyscale = GREYSCALE;

    struct ascii_t ascii;
    image_to_ascii(&ascii, argv[1], greyscale);

    // Tests
    //test_blank(&h_ascii, h_greyscale);
    test_shrek(&ascii);
    
    cleanup_ascii(&ascii);

    return 0;
}
