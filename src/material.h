#pragma once

#include "vec3.h"
#include "ray.h"


class Material {
public:
    char type; // 0 = diffuse, 1 = metal, 2 = dielectric, 3 = emissive, 4 = ...
    int albedo;
    float roughness;
    float metallic;
    float ior;
    float specular;
    float emission;
    int image1_offset;
    int image2_offset;
    int image3_offset;
    unsigned short image_width;
    unsigned short image_height;

    __host__ __device__ Material()
        : albedo(0xFFFFFF),
        roughness(0.0f),
        metallic(0.0f),
        ior(1.5f),
        specular(0.5f),
        image_offset(-1),
        image_width(0),
        image_height(0)
     {}

    // __device__ Vec3 reflect(Ray& ray, const Vec3& normal, const Vec3& pos, curandState* randState) const
    // {
    //     if(curand_uniform(randState) < 0.1f) { // clear coat reflection
    //         ray.position = pos;
    //         Vec3 newDir = ray.direction.reflect(normal.normalize());
    //         ray.direction = newDir;
    //         ray.marchForward(0.0001);
    //         return Vec3(0.95f, 0.95f, 0.95f);

    //     } 
        
    //     // diffuse reflection
    //     ray.position = pos;
    //     Vec3 randVec = Vec3(
    //         curand_normal(randState),
    //         curand_normal(randState),
    //         curand_normal(randState)
    //     );
    //     Vec3 newDir = ray.direction.reflect((normal + randVec * 0.2).normalize());
    //     ray.direction = newDir;
        
    //     ray.marchForward(0.0001);
    //     return Vec3(0.1f, 0.35f, 0.1f);
    // };

    __device__ Vec3 reflect(Ray& ray, const Vec3& normal, const Vec3& pos) const;

    Vec3 color = Vec3(1.0, 1.0, 1.0);
    double roughness = 0.0;
    
};