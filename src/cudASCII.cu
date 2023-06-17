#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "image.h"
#include "constants.h"

__global__ void ascii(char *greyscale, struct image_t *image, char *ascii_value) {

        int luminance = 0; // temp
        ascii_value[threadIdx.x * image->scale_width + blockIdx.x] = greyscale[luminance];

        return;
}

int main(int argc, char **argv) {

        if (argc < 2) {
                printf("Did not provide a filepath\n");
                return -1;
        }

        // Initial data on host
        const char    *h_greyscale = GREYSCALE;

        struct image_t h_image;
        read_image(&h_image, argv[1]);
        char *h_ascii_value = (char*)malloc(h_image.scale_width * h_image.scale_height);

        // Pointers to data on device
        char           *d_greyscale;
        struct image_t *d_image;
        char           *d_ascii_value;

        // Copy greyscale data over
        cudaMallocHost(&d_greyscale, NUM_ASCII);
        cudaMemcpy(d_greyscale, h_greyscale , NUM_ASCII, cudaMemcpyHostToDevice);

        // Copy image struct over
        cudaMallocHost(&d_image, sizeof(struct image_t));
        cudaMemcpy(d_image, &h_image, sizeof(h_image), cudaMemcpyHostToDevice);

        // Copy image data over
        cudaMallocHost(&(d_image->data), d_image->src_width * d_image->src_height);
        cudaMemcpy(d_image->data, h_image.data, d_image->src_width * d_image->src_height, cudaMemcpyHostToDevice);

        // Create memory on for ascii value
        cudaMallocHost(&d_ascii_value, d_image->scale_width * d_image->scale_height);

        // Run the kernel
        ascii<<<h_image.scale_width, h_image.scale_height>>>(d_greyscale, d_image, d_ascii_value);

        // Copy data from device to host
        cudaMemcpy(h_ascii_value, d_ascii_value, h_image.scale_width * h_image.scale_height, cudaMemcpyDeviceToHost);

        for (int i = 0; i < h_image.scale_width * h_image.scale_height; ++i) {
                if (h_ascii_value[i] != h_greyscale[0]) {
                        printf("Error! Did not copy over properly!\nIndex %d has value %d\n", i, h_ascii_value[i]);
                        return -1;
                }
        }

        printf("All values were copied over successfully!\n");

        // Clean up memory
        cudaFreeHost(d_ascii_value);
        cudaFreeHost(d_image->data);
        cudaFreeHost(d_image);
        cudaFreeHost(d_greyscale);
        free(h_ascii_value);
        cleanup_image(&h_image);
        
        cudaDeviceReset();

        return 0;
}
