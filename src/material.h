#pragma once

#include "vec3.h"
#include "ray.h"


class Material {
public:
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