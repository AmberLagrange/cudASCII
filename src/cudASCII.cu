#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "constants.h"

__global__ void ascii(char *greyscale, char *picture_data, char *ascii_value) {

        int luminance = 0; // temp
        ascii_value[threadIdx.x * WIDTH + blockIdx.x] = greyscale[luminance];

        return;
}

int main() {

        // Initial data on host
        const char *h_greyscale                           = GREYSCALE;
        char       h_picture_data[SRC_WIDTH * SRC_HEIGHT] = { 0 };
        char       h_ascii_value[WIDTH * HEIGHT]          = { 0 };

        // Pointers to data on device
        char *d_greyscale;
        char *d_picture_data;
        char *d_ascii_value;

        cudaMalloc(&d_greyscale   , NUM_ASCII);
        cudaMalloc(&d_picture_data, sizeof(h_picture_data));
        cudaMalloc(&d_ascii_value , sizeof(h_ascii_value));

        // Copy data from host to device
        cudaMemcpy(d_greyscale   , h_greyscale   , NUM_ASCII             , cudaMemcpyHostToDevice);
        cudaMemcpy(d_picture_data, h_picture_data, sizeof(h_picture_data), cudaMemcpyHostToDevice);

        ascii<<<WIDTH, HEIGHT>>>(d_greyscale, d_picture_data, d_ascii_value);

        // Copy data from device to host
        cudaMemcpy(h_ascii_value, d_ascii_value, sizeof(h_ascii_value), cudaMemcpyDeviceToHost);

        // Clean up memory
        cudaFree(d_ascii_value);
        cudaFree(d_picture_data);
        cudaFree(d_greyscale);

        for (int i = 0; i < sizeof(h_ascii_value); ++i) {
                if (h_ascii_value[i] != h_greyscale[0]) {
                        printf("Error! Did not copy over properly!\nIndex %d has value %d\n", i, h_ascii_value[i]);
                        return -1;
                }
        }

        printf("All values were copied over successfully!\n");

        return 0;
}
