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

__global__ void convert_to_ascii(ascii_t *ascii, image_t *image) {

    int y_average = 0;
    int u_average = 0;
    int v_average = 0;

    int row = threadIdx.x;
    int col = blockIdx.x;

    int red, green, blue;

    for (int r = row * ascii->scale_height; r < (row + 1) * ascii->scale_height; ++r) {
        for (int c = col * ascii->scale_width; c < (col + 1) * ascii->scale_width; ++c) {
            red   = image->data[(r * image->width + c) * image->bytes_per_pixel + 0];
            green = image->data[(r * image->width + c) * image->bytes_per_pixel + 1];
            blue  = image->data[(r * image->width + c) * image->bytes_per_pixel + 2];

            y_average += RGB_TO_Y(red, green, blue);
            u_average += RGB_TO_U(red, green, blue);
            v_average += RGB_TO_V(red, green, blue);
        }
    }

    y_average /= (ascii->scale_width * ascii->scale_height);
    u_average /= (ascii->scale_width * ascii->scale_height);
    v_average /= (ascii->scale_width * ascii->scale_height);

    y_average = (ascii->dark_mode) ? (255 - y_average) : y_average;

    ascii->y_data[row * ascii->width + col] = ascii->char_set[(y_average * (ascii->char_set_size - 1)) / 255];

    if (ascii->color_enabled) {
        ascii->u_data[row * ascii->width + col] = u_average;
        ascii->v_data[row * ascii->width + col] = v_average;
    }

    return;
}

__host__ int test_blank(ascii_t *ascii, const char *greyscale) {

    char blank_character = ascii->dark_mode ? greyscale[0] : greyscale[strlen(greyscale) - 1];

    for (int i = 0; i < ascii->data_size; ++i) {
        if (ascii->y_data[i] != greyscale[0]) {
            printf("Error! Did not copy over properly!\nIndex %d has value %d\n", i, ascii->y_data[i]);
            return E_TEST;
        }
    }

    printf("All values were copied over successfully!\n");

    return E_OK;
}

__host__ int write_image(ascii_t *ascii, const char *filepath) {

    FILE *file = fopen(filepath, "w+");

    if (!file) {
        fprintf(stderr, "Could not write image to file %s: %s\n", filepath, strerror(errno));
        return E_FILE;
    }

    for (int row = ascii->height - 1; row >= 0 ; --row) {
        for (int col = 0; col < ascii->width; ++col) {
            fprintf(file, "%c", ascii->y_data[row * ascii->width + col]);
        }
        fprintf(file, "\n");
    }

    fclose(file);

    return E_OK;
}

__host__ int image_to_ascii(ascii_t *h_ascii, const char *filepath) {

    // Host image data
    image_t h_image;
    read_image(&h_image, filepath);

    // Host ascii data
    init_ascii(h_ascii, h_image.width, SCALE_WIDTH, h_image.height, SCALE_HEIGHT, 1);
    enable_color(h_ascii);

    // Pointers to data on device
    image_t *d_image;
    ascii_t *d_ascii;

    // Create and copy image struct over
    gpu_check_error(cudaMallocHost(&d_image, sizeof(image_t)));
    gpu_check_error(cudaMemcpy(d_image, &h_image, sizeof(h_image), cudaMemcpyHostToDevice));

    // Create and copy image data over
    gpu_check_error(cudaMallocHost(&(d_image->data), d_image->data_size));
    gpu_check_error(cudaMemcpy(d_image->data, h_image.data, d_image->data_size, cudaMemcpyHostToDevice));

    // Create and copy ascii struct data
    gpu_check_error(cudaMallocHost(&d_ascii, sizeof(ascii_t)));
    gpu_check_error(cudaMemcpy(d_ascii, h_ascii, sizeof(ascii_t), cudaMemcpyHostToDevice));

    // Create and copy ascii char set data
    gpu_check_error(cudaMallocHost(&(d_ascii->char_set), d_ascii->char_set_size));
    gpu_check_error(cudaMemcpy((char*)d_ascii->char_set, h_ascii->char_set, d_ascii->char_set_size, cudaMemcpyHostToDevice));

    // Create ascii data
    gpu_check_error(cudaMallocHost(&(d_ascii->y_data), d_ascii->data_size));

    if (d_ascii->color_enabled) {
        gpu_check_error(cudaMallocHost(&(d_ascii->u_data), d_ascii->data_size));
        gpu_check_error(cudaMallocHost(&(d_ascii->v_data), d_ascii->data_size));
    }

    // Run the kernel
    convert_to_ascii<<<d_ascii->width, d_ascii->height>>>(d_ascii, d_image);
    gpu_check_error(cudaPeekAtLastError());

    // Copy ascii data from device to host
    gpu_check_error(cudaMemcpy(h_ascii->y_data, d_ascii->y_data, h_ascii->data_size, cudaMemcpyDeviceToHost));

    if (d_ascii->color_enabled) {
        gpu_check_error(cudaMemcpy(h_ascii->u_data, d_ascii->u_data, h_ascii->data_size, cudaMemcpyDeviceToHost));
        gpu_check_error(cudaMemcpy(h_ascii->v_data, d_ascii->v_data, h_ascii->data_size, cudaMemcpyDeviceToHost));
    }
    
    // Clean up cuda memory
    gpu_check_error(cudaFreeHost(d_ascii->v_data));
    gpu_check_error(cudaFreeHost(d_ascii->u_data));
    gpu_check_error(cudaFreeHost(d_ascii->y_data));
    gpu_check_error(cudaFreeHost((char*)d_ascii->char_set));
    gpu_check_error(cudaFreeHost(d_ascii));
    gpu_check_error(cudaFreeHost(d_image->data));
    gpu_check_error(cudaFreeHost(d_image));

    cudaDeviceReset();

    // Clean up host memory
    cleanup_image(&h_image);

    return E_OK;
}

int main(int argc, char **argv) {

    if (argc < 2) {
        fprintf(stderr, "Did not provide an image filepath!\n");
        return E_FILE;
    }

    if (argc < 3) {
        fprintf(stderr, "Did not provide a filepath to write to!\n");
        return E_FILE;
    }

    ascii_t ascii;

    image_to_ascii(&ascii, argv[1]);

    if (int ret = write_image(&ascii, argv[2])) {
        return ret;
    }

    cleanup_ascii(&ascii);

    return E_OK;
}
