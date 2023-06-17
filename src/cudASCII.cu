#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "image.h"
#include "constants.h"

__global__ void ascii(char *greyscale, char *image_data, char *ascii_value) {

        int luminance = 0; // temp
        ascii_value[threadIdx.x * WIDTH + blockIdx.x] = greyscale[luminance];

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
        char           h_ascii_value[WIDTH * HEIGHT] = { 0 };

        read_image(&h_image, argv[1]);

        // Pointers to data on device
        char *d_greyscale;
        char *d_image_data;
        char *d_ascii_value;

        cudaMallocHost(&d_greyscale   , NUM_ASCII);
        cudaMallocHost(&d_image_data, sizeof(h_image.data));
        cudaMallocHost(&d_ascii_value , sizeof(h_ascii_value));

        // Copy data from host to device
        cudaMemcpy(d_greyscale   , h_greyscale , NUM_ASCII           , cudaMemcpyHostToDevice);
        cudaMemcpy(d_image_data  , h_image.data, sizeof(h_image.data), cudaMemcpyHostToDevice);

        ascii<<<WIDTH, HEIGHT>>>(d_greyscale, d_image_data, d_ascii_value);

        // Copy data from device to host
        cudaMemcpy(h_ascii_value, d_ascii_value, sizeof(h_ascii_value), cudaMemcpyDeviceToHost);

        // Clean up memory
        cudaFreeHost(d_ascii_value);
        cudaFreeHost(d_image_data);
        cudaFreeHost(d_greyscale);

        for (int i = 0; i < sizeof(h_ascii_value); ++i) {
                if (h_ascii_value[i] != h_greyscale[0]) {
                        printf("Error! Did not copy over properly!\nIndex %d has value %d\n", i, h_ascii_value[i]);
                        return -1;
                }
        }

        printf("All values were copied over successfully!\n");

        return 0;
}
