#include <iostream>
#include <cuda_runtime.h>

#include "texture.h"
#include "vec3.h"
#include "config.h"


// __global__ void swap_channels(float* img, int width, int height, int channels) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x < width && y < height) {
//         int idx = (y * width + x) * channels;
//         float r = img[idx + 0];
//         float g = img[idx + 1];
//         float b = img[idx + 2];

//         // Swap R and B
//         img[idx + 0] = b;
//         img[idx + 1] = g;
//         img[idx + 2] = r;
//     }
// }

__global__ void render(float* pixels) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Vec3 up(0.0, 1.0, 0.0);
    // Vec3 right(1.0, 0.0, 0.0);
    // Vec3::normalize(up);
    // Vec3::normalize(right);
    // Vec3 forward = up.cross(right).normalize();


    if (x < IMAGE_WIDTH && y < IMAGE_WIDTH) {
        int idx = (y * IMAGE_WIDTH + x) * 3;

        pixels[idx + 0] = static_cast<float>(x) / IMAGE_WIDTH;
        pixels[idx + 1] = static_cast<float>(y) / IMAGE_WIDTH;
        pixels[idx + 2] = static_cast<float>(y) / IMAGE_WIDTH;
    }

}



int main(int argc, char** argv) {

    // Texture t("C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\textures\\img256.jpg");

    // size_t img_bytes = t.sizeBytes();
    size_t img_bytes = IMAGE_WIDTH * IMAGE_WIDTH * 3 * sizeof(float);
    float* pixels;
    cudaMalloc(&pixels, img_bytes);
    // cudaMemcpy(d_img, t.getData(), img_bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((IMAGE_WIDTH + block.x - 1) / block.x,
              (IMAGE_WIDTH + block.y - 1) / block.y);

    render<<<grid, block>>>(pixels);
    cudaDeviceSynchronize();

    float* pixels_cpu = new float[IMAGE_WIDTH * IMAGE_WIDTH * 3];
    cudaMemcpy(pixels_cpu, pixels, img_bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);


    Texture::saveImgData("C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\textures\\output.png", pixels_cpu, IMAGE_WIDTH, IMAGE_WIDTH);
    delete[] pixels_cpu;

    // t.save("C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\textures\\output.png");

    // cudaFree(d_img);
    // stbi_image_free(h_img);

    std::cout << "Saved output2\n";
    return 0;
}
