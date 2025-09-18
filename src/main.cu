#include <iostream>
#include <cuda_runtime.h>

#include "texture.h"
#include "vec3.h"
#include "ray.h"
#include "config.h"
#include "sphere.h"
#include "object.h"

#define NUM_OBJECTS 1


// __device__ Vec3 castRay(Ray& ray) {
//     Vec3 color(0.0f, 1.0f, 0.0f);
//     return color;
// }

__device__ void toneMap(Vec3& color) {
    color.x = color.x / (color.x + 1.0f);
    color.y = color.y / (color.y + 1.0f);
    color.z = color.z / (color.z + 1.0f);
}

__global__ void render(float* pixels, Object** objects, float* envTex) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    Vec3 cam_position(0.0f, 0.0f, 0.0f);
    Vec3 up(0.0f, 1.0f, 0.0f);
    Vec3 right(1.0f, 0.0f, 0.0f);
    Vec3::normalize(up);
    Vec3::normalize(right);
    Vec3 forward = up.cross(right).normalize();


    const float sx = static_cast<float>(x + 0.5f) / IMAGE_WIDTH - 0.5f;
    const float sy = static_cast<float>(y + 0.5f) / IMAGE_WIDTH - 0.5f;
    Vec3 s = forward * FOCAL_LENGTH + right * sx + up * sy;
    Vec3 dir = (s - cam_position).normalize();


    if (x < IMAGE_WIDTH && y < IMAGE_WIDTH) {
        
        Ray ray(cam_position, dir);
        Vec3 color(0.0f, 0.0f, 0.0f);
        Vec3 diffuseMultiplier(1.0f, 1.0f, 1.0f);

        for (int j = 0; j<MAX_BOUNCES; j++) {
            float minDistSqr = 1e9;
            Vec3 hitPos;
            Vec3 hitNormal;
            for (int i = 0; i < NUM_OBJECTS; i++) {
                Vec3 pos;
                Vec3 normal;
                if (objects[i]->intersection(ray, pos, normal)) {
                    float newDistSqr = (pos - ray.position).lengthSqr();
                    if (newDistSqr < minDistSqr) {
                        minDistSqr = newDistSqr;
                        hitPos = pos;
                        hitNormal = normal;
                    }
                    
                }
            }
            if (minDistSqr < 1e9) { // hit
                diffuseMultiplier = diffuseMultiplier * Vec3(0.9f, 0.7f, 0.8f);
                ray.position = hitPos;
                Vec3 newDir = ray.direction.reflect((hitNormal).normalize());
                ray.direction = newDir;
                ray.marchForward(0.0001);
            } else {
                // hit the emissive backdrop
                const int width = 4096;
                const int height = 2048;
                float backdropX = atan2(ray.direction.z, ray.direction.x) / (2.0f * 3.14159f) + 0.5f;
                float backdropY = asin(ray.direction.y) / (2.0f * 3.14159f) + 0.5f;
                backdropX *= width;
                backdropY *= height;
                int envImgX = static_cast<int>(backdropX) % width; // wrap around
                int envImgY = static_cast<int>(backdropY) % height;
                int envI = (envImgY * width + envImgX) * 3;
                color += Vec3(envTex[envI + 0], envTex[envI + 1], envTex[envI + 2]) * diffuseMultiplier * 1.0f;
                break;
            }
        }
        
        toneMap(color);

        int idx = (y * IMAGE_WIDTH + x) * 3;
        pixels[idx + 0] = color.x;
        pixels[idx + 1] = color.y;
        pixels[idx + 2] = color.z;
    }

}


__global__ void initScene(Object** objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { // TODO: is this check necessary?
        objects[0] = new Sphere(Vec3(0.0f, 0.0f, -5.0f), 2.0f);

    }
}


int main(int argc, char** argv) 
{
    Object **objects;
    cudaMalloc((void **)&objects, NUM_OBJECTS * sizeof(Object*));
    initScene<<<1, 1>>>(objects);

    size_t img_bytes = IMAGE_WIDTH * IMAGE_WIDTH * 3 * sizeof(float);
    float* pixels;
    cudaMalloc(&pixels, img_bytes);
    // cudaMemcpy(d_img, t.getData(), img_bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);


    Texture environmentMap = Texture("C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\textures\\the_sky_is_on_fire_4k.hdr");
    float* gpuEnvTex;
    cudaMalloc(&gpuEnvTex, environmentMap.sizeBytes());
    cudaMemcpy(gpuEnvTex, environmentMap.getData(), environmentMap.sizeBytes(), cudaMemcpyKind::cudaMemcpyHostToDevice);
    std::cout << "env: " << environmentMap.width << "x" << environmentMap.height << "\n";

    dim3 block(16, 16);
    dim3 grid((IMAGE_WIDTH + block.x - 1) / block.x, (IMAGE_WIDTH + block.y - 1) / block.y);
    render<<<grid, block>>>(pixels, objects, gpuEnvTex);
    cudaDeviceSynchronize();

    float* pixels_cpu = new float[IMAGE_WIDTH * IMAGE_WIDTH * 3];
    cudaMemcpy(pixels_cpu, pixels, img_bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);


    Texture::saveImgData("C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\output\\output.png", pixels_cpu, IMAGE_WIDTH, IMAGE_WIDTH);
    delete[] pixels_cpu;

    // t.save("C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\textures\\output.png");

    // cudaFree(d_img);
    // stbi_image_free(h_img);

    std::cout << "Saved output\n";
    return 0;
}
