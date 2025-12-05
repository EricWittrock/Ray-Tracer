#pragma once
#include "vec3.h"

// #ifndef __CUDACC__
// #define __host__
// #define __device__
// #endif

class Ray {
public:
    Vec3 position;
    Vec3 direction;
    Vec3 diffuseMultiplier;

    __host__ __device__ Ray(const Vec3& origin, const Vec3& direction)
        : position(origin), direction(direction), diffuseMultiplier(1.0f, 1.0f, 1.0f) {}

    __host__ __device__ void marchForward(float distance) {
        position += direction * distance;
    }

    __host__ __device__ Ray copy() const {
        Ray r(position, direction);
        r.diffuseMultiplier = diffuseMultiplier;
        return r;
    }
};