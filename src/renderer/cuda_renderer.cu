#include "cuda_renderer.h"

#define gpu_check_error(code) { gpu_assert((code), __FILE__, __LINE__); }

inline __host__ void gpu_assert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

__global__ void render_ascii(ascii_t *ascii, image_t *image, size_t index, volatile int *error) {

    int y_average = 0;
    int r_average = 0;
    int g_average = 0;
    int b_average = 0;

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

    for (int row = (thread_row + index * MAX_THREADS_PER_BLOCK) * ascii->scale_height; row < (thread_row + index * MAX_THREADS_PER_BLOCK + 1) * ascii->scale_height; ++row) {
        for (int col = thread_col * ascii->scale_width; col < (thread_col + 1) * ascii->scale_width; ++col) {
            byte_1 = image->data[(row * image->width + col) * image->bytes_per_pixel + 0];
            byte_2 = image->data[(row * image->width + col) * image->bytes_per_pixel + 1];
            byte_3 = image->data[(row * image->width + col) * image->bytes_per_pixel + 2];

            r_average += byte_3;
            g_average += byte_2;
            b_average += byte_1;

            conversion_fn(&byte_3, &byte_2, &byte_1);

            y_average += byte_3;
        }
    }

    y_average /= (ascii->scale_width * ascii->scale_height);
    y_average = (ascii->dark_mode) ? (255 - y_average) : y_average;

    ascii->data[(thread_row + index * MAX_THREADS_PER_BLOCK) * ascii->width + thread_col] = ascii->char_set[(y_average * (ascii->char_set_size - 1)) / 255];

    if (ascii->color_enabled) {

        r_average /= (ascii->scale_width * ascii->scale_height);
        g_average /= (ascii->scale_width * ascii->scale_height);
        b_average /= (ascii->scale_width * ascii->scale_height);

        ascii->r_data[(thread_row + index * MAX_THREADS_PER_BLOCK) * ascii->width + thread_col] = r_average;
        ascii->g_data[(thread_row + index * MAX_THREADS_PER_BLOCK) * ascii->width + thread_col] = g_average;
        ascii->b_data[(thread_row + index * MAX_THREADS_PER_BLOCK) * ascii->width + thread_col] = b_average;
    }

    return;
}

__host__ int image_to_ascii(ascii_t *h_ascii, const char *filepath) {

    // Host image data
    image_t h_image;
    if (int ret = read_image(&h_image, filepath)) {
        return ret;
    }

    h_ascii->scale_width = h_image.width / h_ascii->width;
    h_ascii->scale_height = h_image.height / h_ascii->height;

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
    gpu_check_error(cudaMemcpy(d_ascii->char_set, h_ascii->char_set, d_ascii->char_set_size, cudaMemcpyHostToDevice));

    // Create ascii data
    gpu_check_error(cudaMallocHost(&(d_ascii->data), d_ascii->data_size));

    if (d_ascii->color_enabled) {
        gpu_check_error(cudaMallocHost(&(d_ascii->r_data), d_ascii->data_size));
        gpu_check_error(cudaMallocHost(&(d_ascii->g_data), d_ascii->data_size));
        gpu_check_error(cudaMallocHost(&(d_ascii->b_data), d_ascii->data_size));
    }

    // Run the kernel

    // First set of threads that are of size MAX_THREADS_PER_BLOCK (1024)
    size_t index = 0;
    for (; index < d_ascii->height / MAX_THREADS_PER_BLOCK; ++index) {
        render_ascii<<<d_ascii->width, MAX_THREADS_PER_BLOCK>>>(d_ascii, d_image, index, NULL);
        gpu_check_error(cudaPeekAtLastError());
    }

    // Run the remaining number of threads
    render_ascii<<<d_ascii->width, d_ascii->height % MAX_THREADS_PER_BLOCK>>>(d_ascii, d_image, index, NULL);
    gpu_check_error(cudaPeekAtLastError());

    // Copy ascii data from device to host
    gpu_check_error(cudaMemcpy(h_ascii->data, d_ascii->data, h_ascii->data_size, cudaMemcpyDeviceToHost));

    if (d_ascii->color_enabled) {
        gpu_check_error(cudaMemcpy(h_ascii->r_data, d_ascii->r_data, h_ascii->data_size, cudaMemcpyDeviceToHost));
        gpu_check_error(cudaMemcpy(h_ascii->g_data, d_ascii->g_data, h_ascii->data_size, cudaMemcpyDeviceToHost));
        gpu_check_error(cudaMemcpy(h_ascii->b_data, d_ascii->b_data, h_ascii->data_size, cudaMemcpyDeviceToHost));
    }
    
    // Clean up cuda memory
    gpu_check_error(cudaFreeHost(d_ascii->b_data));
    gpu_check_error(cudaFreeHost(d_ascii->g_data));
    gpu_check_error(cudaFreeHost(d_ascii->r_data));
    gpu_check_error(cudaFreeHost(d_ascii->data));
    gpu_check_error(cudaFreeHost(d_ascii->char_set));
    gpu_check_error(cudaFreeHost(d_ascii));
    gpu_check_error(cudaFreeHost(d_image->data));
    gpu_check_error(cudaFreeHost(d_image));

    cudaDeviceReset();

    // Clean up host memory
    cleanup_image(&h_image);

    return E_OK;
}
