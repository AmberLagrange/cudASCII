#include "cuda_renderer.h"

#define gpu_check_error(code) { gpu_assert((code), __FILE__, __LINE__); }

inline __host__ void gpu_assert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

__global__ void render_ascii(ascii_t *ascii, image_t *image, volatile int *error) {

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

            conversion_fn(&byte_1, &byte_3, &byte_2);

            y_average += byte_1;
            u_average += byte_2;
            v_average += byte_3;
        }
    }

    y_average /= (ascii->scale_width * ascii->scale_height);
    y_average = (ascii->dark_mode) ? (255 - y_average) : y_average;

    ascii->data[thread_row * ascii->width + thread_col] = ascii->char_set[(y_average * (ascii->char_set_size - 1)) / 255];

    if (ascii->color_enabled) {

        u_average /= (ascii->scale_width * ascii->scale_height);
        v_average /= (ascii->scale_width * ascii->scale_height);

        ascii->y_data[thread_row * ascii->width + thread_col] = y_average;
        ascii->u_data[thread_row * ascii->width + thread_col] = u_average;
        ascii->v_data[thread_row * ascii->width + thread_col] = v_average;
    }

    return;
}

__host__ int image_to_ascii(ascii_t *h_ascii, const char *filepath) {

    // Host image data
    image_t h_image;
    if (int ret = read_image(&h_image, filepath)) {
        return ret;
    }

    // Host ascii data
    init_ascii(h_ascii, h_image.width, SCALE_WIDTH, h_image.height, SCALE_HEIGHT, 0);
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
    gpu_check_error(cudaMemcpy(d_ascii->char_set, h_ascii->char_set, d_ascii->char_set_size, cudaMemcpyHostToDevice));

    // Create ascii data
    gpu_check_error(cudaMallocHost(&(d_ascii->data), d_ascii->data_size));

    if (d_ascii->color_enabled) {
        gpu_check_error(cudaMallocHost(&(d_ascii->y_data), d_ascii->data_size));
        gpu_check_error(cudaMallocHost(&(d_ascii->u_data), d_ascii->data_size));
        gpu_check_error(cudaMallocHost(&(d_ascii->v_data), d_ascii->data_size));
    }

    // Run the kernel
    render_ascii<<<d_ascii->width, d_ascii->height>>>(d_ascii, d_image, NULL);
    gpu_check_error(cudaPeekAtLastError());

    // Copy ascii data from device to host
    gpu_check_error(cudaMemcpy(h_ascii->data, d_ascii->data, h_ascii->data_size, cudaMemcpyDeviceToHost));

    if (d_ascii->color_enabled) {
        gpu_check_error(cudaMemcpy(h_ascii->y_data, d_ascii->y_data, h_ascii->data_size, cudaMemcpyDeviceToHost));
        gpu_check_error(cudaMemcpy(h_ascii->u_data, d_ascii->u_data, h_ascii->data_size, cudaMemcpyDeviceToHost));
        gpu_check_error(cudaMemcpy(h_ascii->v_data, d_ascii->v_data, h_ascii->data_size, cudaMemcpyDeviceToHost));
    }
    
    // Clean up cuda memory
    gpu_check_error(cudaFreeHost(d_ascii->v_data));
    gpu_check_error(cudaFreeHost(d_ascii->u_data));
    gpu_check_error(cudaFreeHost(d_ascii->y_data));
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
