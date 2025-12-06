#pragma once

#include "vec3.h"
#include "ray.h"
#include <curand_kernel.h>


class Material
{
public:
    char type; // 0 = diffuse, 1 = metal, 2 = dielectric, 3 = emissive, 4 = ...
    int color;
    float p1;
    float p2;
    float p3;
    int image1_offset;
    int image2_offset;
    int image3_offset;
    unsigned short image_width;
    unsigned short image_height;

    __host__ __device__ Material()
        : type(0),
          color(0xFFFFFF),
          p1(0.0f),
          p2(0.0f),
          p3(0.0f),
          image1_offset(-1),
          image2_offset(-1),
          image3_offset(-1),
          image_width(0),
          image_height(0)
    {
    }


    // return true if terminated
    __device__ bool reflect(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, const Material &material, curandState* randState) const
    {
        switch (type) {
            case 0:
                reflectType0(ray, normal, hitPos, material, randState);
                break;
            case 1:
                reflectType1(ray, normal, hitPos, material, randState);
                return true; // terminate after emissive hit
            default:
                return true;
                
        }
        
        return false;
    };

private:

    __device__ void reflectType0(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, const Material &material, curandState* randState) const 
    {
        Vec3 color(0.0f, 1.0f, 0.0f);
        
        if(curand_uniform(randState) < 0.1f) { // clear coat reflection
            ray.diffuseMultiplier = ray.diffuseMultiplier * Vec3(0.95f, 0.95f, 0.95f);
            ray.position = hitPos;
            ray.direction.reflect(normal);
        }
        else { // diffuse reflection
            ray.diffuseMultiplier = ray.diffuseMultiplier * color;
            ray.position = hitPos;
            Vec3 randVec = Vec3(
                curand_normal(randState),
                curand_normal(randState),
                curand_normal(randState)
            );
            ray.direction.reflect((normal + randVec * 0.1f).normalize());
        }
        
        ray.marchForward(0.0001f);
    }

    // emissive
    __device__ void reflectType1(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, const Material &material, curandState* randState) const 
    {
        Vec3 color(0.0f, 3.0f, 3.0f);
        ray.emission = color;
    }
};