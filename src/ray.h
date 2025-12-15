#pragma once
#include "vec3.h"
#include <curand_kernel.h>


// #ifndef __CUDACC__
// #define __host__
// #define __device__
// #endif

class Ray {
public:
    Vec3 position;
    Vec3 direction;
    Vec3 diffuseMultiplier;
    Vec3 emission;
    float refractiveIndex;
    float volumeDensity;
    Vec3 volumeColor;
    Vec3 marchVelocity;

    __host__ __device__ Ray(const Vec3& origin, const Vec3& direction)
        : position(origin),
         direction(direction),
         diffuseMultiplier(1.0f, 1.0f, 1.0f),
         emission(0.0f, 0.0f, 0.0f),
         refractiveIndex(1.0f),
         volumeDensity(0.0f),
         volumeColor(1.0f, 1.0f, 1.0f) {
            marchVelocity = direction * 0.1f;
         }

    __host__ __device__ void marchForward(float distance) {
        position += direction * distance;
    }

    __host__ __device__ Ray copy() const {
        Ray r(position, direction);
        r.diffuseMultiplier = diffuseMultiplier;
        r.refractiveIndex = refractiveIndex;
        r.emission = emission;
        r.refractiveIndex = refractiveIndex;
        r.volumeDensity = volumeDensity;
        r.volumeColor = volumeColor;
        r.marchVelocity = marchVelocity;
        return r;
    }
};