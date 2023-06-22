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

inline __attribute__((always_inline)) __device__ void rgb_to_yuv(u8 *byte_1, u8 *byte_2, u8 *byte_3) {

    u8 r = *byte_1;
    u8 g = *byte_2;
    u8 b = *byte_3;

    *byte_1 = RGB_TO_Y(r, g, b);
    *byte_2 = RGB_TO_U(r, g, b);
    *byte_3 = RGB_TO_V(r, g, b);
}

inline __attribute__((always_inline)) __device__ void yuv_to_rgb(u8 *byte_1, u8 *byte_2, u8 *byte_3) {

    u8 y = *byte_1;
    u8 u = *byte_2;
    u8 v = *byte_3;

    *byte_1 = YUV_TO_R(y, u, v);
    *byte_2 = YUV_TO_G(y, u, v);
    *byte_3 = YUV_TO_B(y, u, v);
}

inline __attribute__((always_inline)) __device__ void rgb_to_rgb(u8 *byte_1, u8 *byte_2, u8 *byte_3) {
    // NOP
}

inline __attribute__((always_inline)) __device__ void yuv_to_yuv(u8 *byte_1, u8 *byte_2, u8 *byte_3) {
    // NOP
}

__global__ void convert_to_ascii(ascii_t *ascii, image_t *image, volatile int *error) {

    int y_average = 0;
    int u_average = 0;
    int v_average = 0;

    int thread_row = threadIdx.x;
    int thread_col = blockIdx.x;

    u8 byte_1, byte_2, byte_3;

    void (*conversion_fn)(u8*, u8*, u8*);

    switch (image->color_format) {
        case COLOR_RGB:
            conversion_fn = &rgb_to_yuv;
            break;
        case COLOR_YUV:
            conversion_fn = &yuv_to_yuv;
            break;
        default:
            if (error) {
                *error = E_INVALID_COLOR_FORMAT;
            }
            return;
    }

    for (int row = thread_row * ascii->scale_height; row < (thread_row + 1) * ascii->scale_height; ++row) {
        for (int col = thread_col * ascii->scale_width; col < (thread_col + 1) * ascii->scale_width; ++col) {
            byte_1 = image->data[(row * image->width + col) * image->bytes_per_pixel + 0];
            byte_2 = image->data[(row * image->width + col) * image->bytes_per_pixel + 1];
            byte_3 = image->data[(row * image->width + col) * image->bytes_per_pixel + 2];

            conversion_fn(&byte_1, &byte_2, &byte_3);

            y_average += byte_1;
            u_average += byte_2;
            v_average += byte_3;
        }
    }

    y_average /= (ascii->scale_width * ascii->scale_height);
    u_average /= (ascii->scale_width * ascii->scale_height);
    v_average /= (ascii->scale_width * ascii->scale_height);

    y_average = (ascii->dark_mode) ? (255 - y_average) : y_average;

    ascii->y_data[thread_row * ascii->width + thread_col] = ascii->char_set[(y_average * (ascii->char_set_size - 1)) / 255];

    if (ascii->color_enabled) {
        ascii->u_data[thread_row * ascii->width + thread_col] = u_average;
        ascii->v_data[thread_row * ascii->width + thread_col] = v_average;
    }

    return;
}

__host__ int test_blank(ascii_t *ascii, const char *greyscale) {

    char blank_character = ascii->dark_mode ? greyscale[0] : greyscale[strlen(greyscale) - 1];

    for (int i = 0; i < ascii->data_size; ++i) {
        if (ascii->y_data[i] != greyscale[0]) {
            printf("Error! Did not copy over properly!\nIndex %d has value %d\n", i, ascii->y_data[i]);
            return E_TEST_FAILED;
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
    convert_to_ascii<<<d_ascii->width, d_ascii->height>>>(d_ascii, d_image, NULL);
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
