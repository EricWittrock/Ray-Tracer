#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "texture.h"
#include "vec3.h"
#include "ray.h"
#include "config.h"
#include "sphere.h"
#include "floor.h"
#include "object.h"
#include "model.h"

// __device__ Vec3 castRay(Ray& ray) {
//     Vec3 color(0.0f, 1.0f, 0.0f);
//     return color;
// }

__device__ void toneMap(Vec3& color) {
    // exposure
    const float exposure = -0.8f;
    const float exposure_factor = powf(2.0f, exposure);

    // ACES tone mapping
    color.x = color.x * (color.x * 2.51f + 0.03f) / (color.x * (color.x * 2.43f + 0.59f) + 0.14f) * exposure_factor;
    color.y = color.y * (color.y * 2.51f + 0.03f) / (color.y * (color.y * 2.43f + 0.59f) + 0.14f) * exposure_factor;
    color.z = color.z * (color.z * 2.51f + 0.03f) / (color.z * (color.z * 2.43f + 0.59f) + 0.14f) * exposure_factor;

    // gamma correction
    const float inv_gamma = 1.0f / 2.2f;
    color.x = powf(color.x, inv_gamma);
    color.y = powf(color.y, inv_gamma);
    color.z = powf(color.z, inv_gamma);
}

__global__ void render(float* pixels, Object** objects, float* envTex) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= IMAGE_WIDTH || y >= IMAGE_WIDTH) return;

    curandState randState;
    curand_init(37811, x + y * IMAGE_WIDTH, 0, &randState);

    Vec3 cam_position(0.0f, 0.0f, 0.0f);
    Vec3 up(0.0f, 1.0f, 0.0f);
    Vec3 right(1.0f, 0.0f, 0.0f);
    Vec3::normalize(up);
    Vec3::normalize(right);
    Vec3 forward = up.cross(right).normalize();


    const float sx = static_cast<float>(x + 0.5f) / IMAGE_WIDTH - 0.5f;
    const float sy = 0.5f - static_cast<float>(y + 0.5f) / IMAGE_WIDTH;
    Vec3 s = forward * FOCAL_LENGTH + right * sx + up * sy;
    Vec3 dir = (s - cam_position).normalize();

    Vec3 color(0.0f, 0.0f, 0.0f);

    for (int k = 0; k < NUM_SAMPLES; k++) {
        Ray ray(cam_position, dir);
        Vec3 diffuseMultiplier(1.0f, 1.0f, 1.0f);

        for (int j = 0; j<MAX_BOUNCES; j++) {
            float minDistSqr = 1e9;
            Vec3 hitPos;
            Vec3 hitNormal;
            Object *hitObject = nullptr;
            for (int i = 0; i < NUM_OBJECTS; i++) {
                Vec3 pos;
                Vec3 normal;
                if (objects[i]->intersection(ray, pos, normal)) {
                    float newDistSqr = (pos - ray.position).lengthSqr();
                    if (newDistSqr < minDistSqr) {
                        minDistSqr = newDistSqr;
                        hitPos = pos;
                        hitNormal = normal;
                        hitObject = objects[i];
                    }
                }
            }
            if (hitObject != nullptr) { // hit
                // hitObject->material.reflect(ray, hitNormal, hitPos, &randState);
                // hitObject->material.reflect(ray, hitNormal, hitPos);

                
                if(curand_uniform(&randState) < 0.1f) { // clear coat reflection
                    diffuseMultiplier = diffuseMultiplier * Vec3(0.95f, 0.95f, 0.95f);
                    ray.position = hitPos;
                    Vec3 newDir = ray.direction.reflect((hitNormal).normalize());
                    ray.direction = newDir;

                } else { // diffuse reflection
                    diffuseMultiplier = diffuseMultiplier * Vec3(0.1f, 0.35f, 0.1f);
                    ray.position = hitPos;
                    Vec3 randVec = Vec3(
                        curand_normal(&randState),
                        curand_normal(&randState),
                        curand_normal(&randState)
                    );
                    Vec3 newDir = ray.direction.reflect((hitNormal + randVec * 0.2).normalize());
                    ray.direction = newDir;
                }
                
                // ray.marchForward(0.0001);
            } else {
                // hit the emissive backdrop
                const int width = 4096; // TODO: don't hardcode dimensions
                const int height = 2048;
                float backdropX = atan2(ray.direction.z, ray.direction.x) / (2.0f * 3.14159f) + 0.5f;
                float backdropY = -asin(ray.direction.y) / (2.0f * 3.14159f) + 0.5f;
                backdropX *= width;
                backdropY *= height;
                int envImgX = static_cast<int>(backdropX) % width; // wrap around
                int envImgY = static_cast<int>(backdropY) % height;
                int envI = (envImgY * width + envImgX) * 3;
                color += Vec3(envTex[envI + 0], envTex[envI + 1], envTex[envI + 2]) * diffuseMultiplier * 1.0f;
                break;
            }
        }
    }

    color /= static_cast<float>(NUM_SAMPLES);
    toneMap(color);

    int idx = (y * IMAGE_WIDTH + x) * 3;
    pixels[idx + 0] = color.x;
    pixels[idx + 1] = color.y;
    pixels[idx + 2] = color.z;
    

}


__device__ bool rayTriangleIntersect(Ray& ray, const Vec3& v0, const Vec3& v1, const Vec3& v2, Vec3& outPos) {
    const float epsilon = 1e-9f;
    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    Vec3 ray_cross_e2 = ray.direction.cross(edge2);

    float det = edge1.dot(ray_cross_e2);
    if (det > -epsilon && det < epsilon) { // parallel
        return false; 
    }

    float inv_det = 1.0f / det;
    Vec3 s = ray.position - v0;
    float u = inv_det * s.dot(ray_cross_e2);
    if (u < 0.0f || u > 1.0f)
        return false;

    Vec3 q = s.cross(edge1);
    float v = inv_det * ray.direction.dot(q);
    if (v < 0.0f || u + v > 1.0f)
        return false;

    float t = inv_det * edge2.dot(q);
    if (t > epsilon) {
        outPos = ray.position + ray.direction * t;
        return true;
    } else {
        return false;
    }
}

__global__ void render2(float* pixels, float* tris, int numTris, float* envTex) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= IMAGE_WIDTH || y >= IMAGE_WIDTH) return;

    curandState randState;
    curand_init(37811, x + y * IMAGE_WIDTH, 0, &randState); // 37811 is a big arbitrary prime

    Vec3 cam_position(0.0f, 0.0f, 0.0f);
    Vec3 up(0.0f, 1.0f, 0.0f);
    Vec3 right(1.0f, 0.0f, 0.0f);
    Vec3::normalize(up);
    Vec3::normalize(right);
    Vec3 forward = up.cross(right).normalize();


    const float sx = static_cast<float>(x + 0.5f) / IMAGE_WIDTH - 0.5f;
    const float sy = 0.5f - static_cast<float>(y + 0.5f) / IMAGE_WIDTH;
    Vec3 s = forward * FOCAL_LENGTH + right * sx + up * sy;
    Vec3 dir = (s - cam_position).normalize();

    Vec3 color(0.0f, 0.0f, 0.0f);

    for (int k = 0; k < NUM_SAMPLES; k++) {
        Ray ray(cam_position, dir);
        Vec3 diffuseMultiplier(1.0f, 1.0f, 1.0f);

        for (int j = 0; j<MAX_BOUNCES; j++) {
            Vec3 hitPos;
            int hitTriIndex = -1;
            float minDistSqr = 1e12;
            for (int i = 0; i < numTris; i++) {
                Vec3 v0(tris[i * 9 + 0], tris[i * 9 + 1], tris[i * 9 + 2]);
                Vec3 v1(tris[i * 9 + 3], tris[i * 9 + 4], tris[i * 9 + 5]);
                Vec3 v2(tris[i * 9 + 6], tris[i * 9 + 7], tris[i * 9 + 8]);
                
                Vec3 pos;
                if (rayTriangleIntersect(ray, v0, v1, v2, pos)) {
                    float newDistSqr = (pos - ray.position).lengthSqr();
                    if (newDistSqr < minDistSqr) {
                        minDistSqr = newDistSqr;
                        hitPos = pos;
                        hitTriIndex = i;
                    }
                }
            }
            if (hitTriIndex >= 0) { // hit
                Vec3 v0(tris[hitTriIndex * 9 + 0], tris[hitTriIndex * 9 + 1], tris[hitTriIndex * 9 + 2]);
                Vec3 v1(tris[hitTriIndex * 9 + 3], tris[hitTriIndex * 9 + 4], tris[hitTriIndex * 9 + 5]);
                Vec3 v2(tris[hitTriIndex * 9 + 6], tris[hitTriIndex * 9 + 7], tris[hitTriIndex * 9 + 8]);
                Vec3 edge1 = v1 - v0;
                Vec3 edge2 = v2 - v0;
                Vec3 hitNormal = edge1.cross(edge2).normalize();
                
                if(curand_uniform(&randState) < 0.1f) { // clear coat reflection
                    diffuseMultiplier = diffuseMultiplier * Vec3(0.95f, 0.95f, 0.95f);
                    ray.position = hitPos;
                    Vec3 newDir = ray.direction.reflect(hitNormal);
                    ray.direction = newDir;

                } else { // diffuse reflection
                    diffuseMultiplier = diffuseMultiplier * Vec3(0.1f, 0.35f, 0.1f);
                    ray.position = hitPos;
                    Vec3 randVec = Vec3(
                        curand_normal(&randState),
                        curand_normal(&randState),
                        curand_normal(&randState)
                    );
                    Vec3 newDir = ray.direction.reflect((hitNormal + randVec * 0.2f).normalize());
                    ray.direction = newDir;
                }
                
                ray.marchForward(0.0001f);
            } else {
                // hit the emissive backdrop
                const int width = 4096; // TODO: don't hardcode dimensions
                const int height = 2048;
                float backdropX = atan2(ray.direction.z, ray.direction.x) / (2.0f * 3.14159f) + 0.5f;
                float backdropY = -asin(ray.direction.y) / (2.0f * 3.14159f) + 0.5f;
                backdropX *= width;
                backdropY *= height;
                int envImgX = static_cast<int>(backdropX) % width; // wrap around
                int envImgY = static_cast<int>(backdropY) % height;
                int envI = (envImgY * width + envImgX) * 3;
                color += Vec3(envTex[envI + 0], envTex[envI + 1], envTex[envI + 2]) * diffuseMultiplier * 1.0f;
                break;
            }
        }
    }

    color /= static_cast<float>(NUM_SAMPLES);
    toneMap(color);

    int idx = (y * IMAGE_WIDTH + x) * 3;
    pixels[idx + 0] = color.x;
    pixels[idx + 1] = color.y;
    pixels[idx + 2] = color.z;
    

}


__global__ void initScene(Object** objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { // TODO: is this check necessary?
        objects[0] = new Sphere(Vec3(0.0f, -1.2f, -5.0f), 2.0f);
        objects[1] = new Sphere(Vec3(3.0f, 3.0f, -5.4f), 1.5f);
        objects[2] = new Floor(-3.2f);
    }
}

void buildScene(Object** objects) {
    objects[0] = new Sphere(Vec3(0.0f, -1.2f, -5.0f), 2.0f);
    objects[1] = new Sphere(Vec3(3.0f, 3.0f, -5.4f), 1.5f);
    

    // objects[2] = &m;
}

void loadAssets() {

}


int main(int argc, char** argv) 
{

    Model model;
    model.loadMesh("C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\textures\\monkey.obj", Vec3(-1.5f, -1.0f, -3.0f));

    float *gpuTris;
    cudaMalloc((void **)&gpuTris, model.getSizeBytes());
    cudaMemcpy(gpuTris, model.getFaces(), model.getSizeBytes(), cudaMemcpyKind::cudaMemcpyHostToDevice);
    int numTris = model.getNumTris();

    Object **objects;
    cudaMalloc((void **)&objects, NUM_OBJECTS * sizeof(Object*));
    initScene<<<1, 1>>>(objects);

    size_t img_bytes = IMAGE_WIDTH * IMAGE_WIDTH * 3 * sizeof(float);
    float* pixels;
    cudaMalloc(&pixels, img_bytes);

    Texture environmentMap = Texture("C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\textures\\the_sky_is_on_fire_4k.hdr");
    float* gpuEnvTex;
    cudaMalloc(&gpuEnvTex, environmentMap.sizeBytes());
    cudaMemcpy(gpuEnvTex, environmentMap.getData(), environmentMap.sizeBytes(), cudaMemcpyKind::cudaMemcpyHostToDevice);
    std::cout << "env: " << environmentMap.width << "x" << environmentMap.height << "\n";

    dim3 block(16, 16);
    dim3 grid((IMAGE_WIDTH + block.x - 1) / block.x, (IMAGE_WIDTH + block.y - 1) / block.y);
    render2<<<grid, block>>>(pixels, gpuTris, numTris, gpuEnvTex);
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
