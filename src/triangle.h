#pragma once
// #include <cmath>
#include "vec3.h"

// #ifndef __CUDACC__
// #define __host__
// #define __device__
// #endif


class Triangle {
public:
    Vec3 v0, v1, v2;
    __host__ __device__ Triangle() : v0(), v1(), v2() {}
    __host__ __device__ Triangle(const Vec3& v0, const Vec3& v1, const Vec3& v2) : v0(v0), v1(v1), v2(v2) {}

    __host__ __device__ inline float getArea() const {
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        return edge1.cross(edge2).length() * 0.5f;
    }

    __host__ __device__ inline Vec3 getNormal() const {
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        return edge1.cross(edge2).normalize();
    }

    __host__ __device__ inline void operator = (const Triangle& t) {
        v0 = t.v0;
        v1 = t.v1;
        v2 = t.v2;
    }

    // TODO: static operators to act on an array
};