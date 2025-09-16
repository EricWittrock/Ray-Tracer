#pragma once
#include "vec3.h"

// #ifndef __CUDACC__
// #define __host__
// #define __device__
// #endif

class Ray {
public:
    __host__ __device__ Ray(const Vec3& origin, const Vec3& direction)
        : position(origin), direction(direction) {}

    __host__ __device__ void marchForward(float distance) {
        position += direction * distance;
    }

    __host__ __device__ Ray copy() const {
        Ray r(position, direction);
        r.numBounces = numBounces;
        return r;
    }

    Vec3 position;
    Vec3 direction;
    int numBounces = 0;
};