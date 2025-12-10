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
#include "sceneParser.h"
#include "material.h"
#include "BVHcreator.h"

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


__device__ bool rayTriangleIntersect(Ray& ray, const Vec3& v0, const Vec3& v1, const Vec3& v2, Vec3& outPos) {
    const float epsilon = 1e-9f;
    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    Vec3 ray_cross_e2 = ray.direction.cross(edge2);

    float det = edge1.dot(ray_cross_e2);
    if (det > -epsilon && det < epsilon) { // ray is parallel to triangle plane
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

__device__ Vec3 barycentric(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& pos) {
    Vec3 v0 = b - a;
    Vec3 v1 = c - a;
    Vec3 v2 = pos - a;
    float d00 = v0.dot(v0);
    float d01 = v0.dot(v1);
    float d11 = v1.dot(v1);
    float d20 = v2.dot(v0);
    float d21 = v2.dot(v1);
    float denom = d00 * d11 - d01 * d01;
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;
    return Vec3(u, v, w);
}

__device__ Vec3 interpolateNormal(const Vec3& n0, const Vec3& n1, const Vec3& n2, const Vec3& bary) {
    return (n0 * bary.x + n1 * bary.y + n2 * bary.z).normalize();
}


__device__ float AABBIntersectDistance(const Ray& ray, const BVH::BVHNode& node) {
    // Check if origin is inside the AABB
    if (ray.position.x >= node.bbox_min_x && ray.position.x <= node.bbox_max_x &&
        ray.position.y >= node.bbox_min_y && ray.position.y <= node.bbox_max_y &&
        ray.position.z >= node.bbox_min_z && ray.position.z <= node.bbox_max_z)
    {
        return 0.0f;
    }

    float tmin = 0.0f;
    float tmax = 1e13f;
    float invDirX = 1.0f / ray.direction.x;
    float invDirY = 1.0f / ray.direction.y;
    float invDirZ = 1.0f / ray.direction.z;

    float t1;
    float t2;
    float tNear;
    float tFar;

    t1 = (node.bbox_min_x - ray.position.x) * invDirX;
    t2 = (node.bbox_max_x - ray.position.x) * invDirX;
    tNear = fminf(t1, t2);
    tFar = fmaxf(t1, t2);
    tmin = fmaxf(tmin, tNear);
    tmax = fminf(tmax, tFar);

    t1 = (node.bbox_min_y - ray.position.y) * invDirY;
    t2 = (node.bbox_max_y - ray.position.y) * invDirY;
    tNear = fminf(t1, t2);
    tFar = fmaxf(t1, t2);
    tmin = fmaxf(tmin, tNear);
    tmax = fminf(tmax, tFar);

    t1 = (node.bbox_min_z - ray.position.z) * invDirZ;
    t2 = (node.bbox_max_z - ray.position.z) * invDirZ;
    tNear = fminf(t1, t2);
    tFar = fmaxf(t1, t2);
    tmin = fmaxf(tmin, tNear);
    tmax = fminf(tmax, tFar);

    if (tmax < tmin) return 1e13f;
    if (tmin < 0.0f) return 1e13f;
    return tmin;
}

__global__ void render2(float* pixels, float* tris, int numTris, BVH::BVHNode* bvhNodes, Material *materials, float* envTex) {
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

        for (int j = 0; j<MAX_BOUNCES; j++) {
            Vec3 hitPos;
            int hitTriIndex = -1;
            float minDistSqr = 1e12f;

            ////////////////////////////////////////////////////////////////////////////////
            BVH::BVHNode* stack[BVH_DEPTH];
            int stackPtr = 0;
            // stack[stackPtr++] = &bvhNodes[0];
            if (AABBIntersectDistance(ray, bvhNodes[0]) < 1e12) {
                stack[stackPtr++] = &bvhNodes[0];
            }

            Vec3 numTests(0.0f, 0.0f, 0.0f);
            while(stackPtr > 0) {
                BVH::BVHNode* node = stack[--stackPtr];

                // if (AABBIntersectDistance(ray, *node) > 1e12) {
                //     continue;
                // }

                if (node->childAIndex == -1) { // leaf node
                    int startTriData = node->startTriDataOffs;
                    int endTriData = startTriData + node->triDataLength;
                    for (int i = startTriData; i < endTriData; i+=25) {
                        Vec3 v0(tris[i + 0], tris[i + 1], tris[i + 2]);
                        Vec3 v1(tris[i + 3], tris[i + 4], tris[i + 5]);
                        Vec3 v2(tris[i + 6], tris[i + 7], tris[i + 8]);
                        
                        numTests.x += 0.1f;
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
                } else {
                    // add nearest box last so it gets processed first
                    numTests.y += 1.0f;
                    float distAsqr = AABBIntersectDistance(ray, bvhNodes[node->childAIndex]);
                    float distBsqr = AABBIntersectDistance(ray, bvhNodes[node->childBIndex]);
                    distAsqr = distAsqr * distAsqr;
                    distBsqr = distBsqr * distBsqr;

                    if (distAsqr < distBsqr) {
                        if (distBsqr < minDistSqr) stack[stackPtr++] = &bvhNodes[node->childBIndex];
                        if (distAsqr < minDistSqr) stack[stackPtr++] = &bvhNodes[node->childAIndex];
                    } else {
                        if (distAsqr < minDistSqr) stack[stackPtr++] = &bvhNodes[node->childAIndex];
                        if (distBsqr < minDistSqr) stack[stackPtr++] = &bvhNodes[node->childBIndex];
                    }
                    // stack[stackPtr++] = &bvhNodes[node->childAIndex];
                    // stack[stackPtr++] = &bvhNodes[node->childBIndex];
                }
            }

            // for (int i = 0; i < numTris; i+=25) {
            //     Vec3 v0(tris[i + 0], tris[i + 1], tris[i + 2]);
            //     Vec3 v1(tris[i + 3], tris[i + 4], tris[i + 5]);
            //     Vec3 v2(tris[i + 6], tris[i + 7], tris[i + 8]);
                
            //     Vec3 pos;
            //     if (rayTriangleIntersect(ray, v0, v1, v2, pos)) {
            //         float newDistSqr = (pos - ray.position).lengthSqr();
            //         if (newDistSqr < minDistSqr) {
            //             minDistSqr = newDistSqr;
            //             hitPos = pos;
            //             hitTriIndex = i;
            //         }
            //     }
            // }
            ////////////////////////////////////////////////////////////////////////////////
            // color += numTests * 0.01;
            // break;

            if (hitTriIndex >= 0) { // hit
                Vec3 v0(tris[hitTriIndex + 0], tris[hitTriIndex + 1], tris[hitTriIndex + 2]);
                Vec3 v1(tris[hitTriIndex + 3], tris[hitTriIndex + 4], tris[hitTriIndex + 5]);
                Vec3 v2(tris[hitTriIndex + 6], tris[hitTriIndex + 7], tris[hitTriIndex + 8]);

                Vec3 n0(tris[hitTriIndex + 9], tris[hitTriIndex + 10], tris[hitTriIndex + 11]);
                Vec3 n1(tris[hitTriIndex + 12], tris[hitTriIndex + 13], tris[hitTriIndex + 14]);
                Vec3 n2(tris[hitTriIndex + 15], tris[hitTriIndex + 16], tris[hitTriIndex + 17]);
                Vec3 bary = barycentric(v0, v1, v2, hitPos);
                Vec3 hitNormal = interpolateNormal(n0, n1, n2, bary);

                int hitIndex = static_cast<int>(tris[hitTriIndex + 24]);
                Material hitMaterial = materials[hitIndex];

                bool terminated = hitMaterial.reflect(ray, hitNormal, hitPos, hitMaterial, &randState);
                if (terminated) {
                    color += ray.emission * ray.diffuseMultiplier;
                    break;
                }
                Vec3::normalize(ray.direction);

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
                color += Vec3(envTex[envI + 0], envTex[envI + 1], envTex[envI + 2]) * ray.diffuseMultiplier * BACKGROUND_BRIGHTNESS;
                break;
            }
        }
    }

    color /= static_cast<float>(NUM_SAMPLES);
    toneMap(color);
    // if (color.x > 1.0f) color.x = 1.0f;
    // if (color.y > 1.0f) color.y = 1.0f;
    // if (color.z > 1.0f) color.z = 1.0f;
    // if (color.x < 0.0f) color.x = 0.0f;
    // if (color.y < 0.0f) color.y = 0.0f;
    // if (color.z < 0.0f) color.z = 0.0f;

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
    SceneParser parser;
    parser.parseFromFile("C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\scene.txt");

    float *cpuTris;
    size_t numTris;
    BVH::BVHNode* bvh_nodes = nullptr;
    int num_bvh_nodes = 0;
    float *gpuTris;
    parser.getTriangleData(&cpuTris, &numTris, &bvh_nodes, &num_bvh_nodes);
    cudaMalloc((void **)&gpuTris, numTris * sizeof(float));
    cudaMemcpy(gpuTris, cpuTris, numTris * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
    BVH::BVHNode* gpuBVHNodes;
    cudaMalloc((void **)&gpuBVHNodes, num_bvh_nodes * sizeof(BVH::BVHNode));
    cudaMemcpy(gpuBVHNodes, bvh_nodes, num_bvh_nodes * sizeof(BVH::BVHNode), cudaMemcpyKind::cudaMemcpyHostToDevice);

    const Material *materials;
    size_t numMaterials;
    parser.getMaterialData(&materials, &numMaterials);
    Material *gpuMaterials;
    cudaMalloc((void **)&gpuMaterials, numMaterials * sizeof(Material));
    cudaMemcpy(gpuMaterials, materials, numMaterials * sizeof(Material), cudaMemcpyKind::cudaMemcpyHostToDevice);

    size_t img_bytes = IMAGE_WIDTH * IMAGE_WIDTH * 3 * sizeof(float);
    float* pixels;
    cudaMalloc(&pixels, img_bytes);

    Texture environmentMap = Texture("C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\textures\\kloofendal_43d_clear_4k.hdr");
    float* gpuEnvTex;
    cudaMalloc(&gpuEnvTex, environmentMap.sizeBytes());
    cudaMemcpy(gpuEnvTex, environmentMap.getData(), environmentMap.sizeBytes(), cudaMemcpyKind::cudaMemcpyHostToDevice);
    std::cout << "env texture size: " << environmentMap.width << "x" << environmentMap.height << "\n";

    dim3 block(16, 16);
    dim3 grid((IMAGE_WIDTH + block.x - 1) / block.x, (IMAGE_WIDTH + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    std::cout << "begin rendering" << std::endl;
    render2<<<grid, block>>>(pixels, gpuTris, static_cast<int>(numTris), gpuBVHNodes, gpuMaterials, gpuEnvTex);
    cudaDeviceSynchronize();
    std::cout << "done rendering" << std::endl;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Render time: " << milliseconds << " ms" << std::endl;

    float* pixels_cpu = new float[IMAGE_WIDTH * IMAGE_WIDTH * 3];
    cudaMemcpy(pixels_cpu, pixels, img_bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);


    Texture::saveImgData("C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\output\\output.png", pixels_cpu, IMAGE_WIDTH, IMAGE_WIDTH);
    delete[] pixels_cpu;


    // cudaFree(d_img);
    // stbi_image_free(h_img);

    std::cout << "Saved output\n";
    return 0;
}
