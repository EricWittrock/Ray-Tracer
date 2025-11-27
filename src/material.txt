#include "material.h"
// #include <curand_kernel.h>

__device__ Vec3 Material::reflect(Ray& ray, const Vec3& normal, const Vec3& pos) const
{
    // if(curand_uniform(randState) < 0.1f) { // clear coat reflection
    //     ray.position = pos;
    //     Vec3 newDir = ray.direction.reflect(normal.normalize());
    //     ray.direction = newDir;
    //     ray.marchForward(0.0001);
    //     return Vec3(0.95f, 0.95f, 0.95f);
    // }

    // // diffuse reflection
    // ray.position = pos;
    // Vec3 randVec = Vec3(
    //     curand_normal(randState),
    //     curand_normal(randState),
    //     curand_normal(randState)
    // );
    // Vec3 newDir = ray.direction.reflect((normal + randVec * 0.2).normalize());
    // ray.direction = newDir;

    // ray.marchForward(0.0001);
    // return Vec3(0.1f, 0.35f, 0.1f);

    // diffuse reflection
    ray.position = pos;
    Vec3 newDir = ray.direction.reflect((normal).normalize());
    ray.direction = newDir;

    ray.marchForward(0.0001);
    return Vec3(0.1f, 0.35f, 0.1f);
}