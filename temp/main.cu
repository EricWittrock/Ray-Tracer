#include <iostream>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image/stb_image_write.h"

// CUDA kernel to swap R and B channels
__global__ void swap_channels(unsigned char* img, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        unsigned char r = img[idx + 0];
        unsigned char g = img[idx + 1];
        unsigned char b = img[idx + 2];

        // Swap R and B
        img[idx + 0] = b;
        img[idx + 1] = g;
        img[idx + 2] = r;
    }
}

int main(int argc, char** argv) {

    int width, height, channels;
    unsigned char* h_img = stbi_load("C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\textures\\img256.jpg", &width, &height, &channels, 0);
    if (!h_img) {
        std::cerr << "Failed to load image\n";
        return 1;
    }

    size_t img_size = width * height * channels;
    unsigned char* d_img;
    cudaMalloc(&d_img, img_size);
    cudaMemcpy(d_img, h_img, img_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    swap_channels<<<grid, block>>>(d_img, width, height, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);

    stbi_write_png("C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\textures\\output.png", width, height, channels, h_img, width * channels);

    cudaFree(d_img);
    stbi_image_free(h_img);

    std::cout << "Saved output\n";
    return 0;
}
