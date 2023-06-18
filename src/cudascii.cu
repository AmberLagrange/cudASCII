#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "constants.h"

#include "image.h"
#include "ascii.h"

__global__ void ascii(char *greyscale, struct image_t *image, struct ascii_t *ascii) {

    int luminance = 0; // temp
    ascii->data[threadIdx.x * ascii->width + blockIdx.x] = greyscale[luminance];

    return;
}

__host__ int test_blank(struct ascii_t *ascii, const char *greyscale) {

    for (int i = 0; i < ascii->width * ascii->height; ++i) {
        if (ascii->data[i] != greyscale[0]) {
            printf("Error! Did not copy over properly!\nIndex %d has value %d\n", i, ascii->data[i]);
            return -1;
        }
    }

    printf("All values were copied over successfully!\n");

    return 0;
}

int main(int argc, char **argv) {

    if (argc < 2) {
        printf("Did not provide a filepath\n");
        return -1;
    }

    // Initial data on host

    // Host greyscale data
    const char *h_greyscale = GREYSCALE;

    // Host image data
    struct image_t h_image;
    read_image(&h_image, argv[1]);

    // Host ascii data
    struct ascii_t h_ascii;
    init_ascii(&h_ascii, h_image.width, SCALE_WIDTH, h_image.height, SCALE_HEIGHT);

    // Pointers to data on device
    char       *d_greyscale;
    struct image_t *d_image;
    struct ascii_t *d_ascii;

    // Create and copy greyscale data over
    cudaMallocHost(&d_greyscale, NUM_ASCII);
    cudaMemcpy(d_greyscale, h_greyscale , NUM_ASCII, cudaMemcpyHostToDevice);

    // Create and copy image struct over
    cudaMallocHost(&d_image, sizeof(struct image_t));
    cudaMemcpy(d_image, &h_image, sizeof(h_image), cudaMemcpyHostToDevice);

    // Create and copy image data over
    cudaMallocHost(&(d_image->data), d_image->width * d_image->height);
    cudaMemcpy(d_image->data, h_image.data, d_image->width * d_image->height, cudaMemcpyHostToDevice);

    // Create and copy ascii struct data
    cudaMallocHost(&d_ascii, sizeof(struct ascii_t));
    cudaMemcpy(d_ascii, &h_ascii, sizeof(struct ascii_t), cudaMemcpyHostToDevice);

    // Create ascii data
    cudaMallocHost(&(d_ascii->data), d_ascii->width * d_ascii->height);

    // Run the kernel
    ascii<<<h_ascii.width, h_ascii.height>>>(d_greyscale, d_image, d_ascii);

    // Copy ascii data from device to host
    cudaMemcpy(h_ascii.data, d_ascii->data, h_ascii.width * h_ascii.height, cudaMemcpyDeviceToHost);

    // Tests
    test_blank(&h_ascii, h_greyscale);

    // Clean up cuda memory
    cudaFreeHost(d_ascii->data);
    cudaFreeHost(d_ascii);
    cudaFreeHost(d_image->data);
    cudaFreeHost(d_image);
    cudaFreeHost(d_greyscale);

    // Clean up host memory
    cleanup_ascii(&h_ascii);
    cleanup_image(&h_image);
    
    cudaDeviceReset();

    return 0;
}
