#pragma once
#include <cmath>
#include <cstdlib>
#include <random>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif


class Vec3 {
public:
    float x, y, z;
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3() : x(0.0f), y(0.0f), z(0.0f) {}

    // __host__ __device__ static Vec3 random() {
    //     return Vec3(
    //         (static_cast<float>(rand()) / RAND_MAX) * 2.0 - 1.0,
    //         (static_cast<float>(rand()) / RAND_MAX) * 2.0 - 1.0,
    //         (static_cast<float>(rand()) / RAND_MAX) * 2.0 - 1.0
    //     );
    // }

    // __host__ __device__ static Vec3 randomGaussian(std::mt19937 &rng, float stddev = 1.0) {
    //     std::normal_distribution<float> dist(0.0, stddev);
    //     return Vec3(dist(rng), dist(rng), dist(rng));
    // }

    __host__ __device__ inline float dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    __host__ __device__ inline Vec3 cross(const Vec3& v) const {
        return Vec3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }

    __host__ __device__ inline float lengthSqr() const {
        return x * x + y * y + z * z;
    }

    __host__ __device__ inline float length() const {
        return std::sqrtf(lengthSqr());
    }

    __host__ __device__ inline Vec3 reflect(const Vec3& normal) const {
        float d = 2.0f * dot(normal);
        return Vec3(
            x - d * normal.x,
            y - d * normal.y,
            z - d * normal.z
        );
        // return this->copy();
    }

    __host__ __device__ inline Vec3 normalize() const {
        float invlen = 1.0f / length();
        if (invlen > 0) {
            return Vec3(x * invlen, y * invlen, z * invlen);
        }
        return Vec3(0, 0, 0);
    }

    __host__ __device__ static Vec3 normalize(Vec3& v) {
        float invlen = 1.0f / v.length();
        if (invlen > 0) {
            v.x *= invlen;
            v.y *= invlen;
            v.z *= invlen;
        }
        return v;
    }

    __host__ __device__ inline Vec3 copy() const {
        return Vec3(x, y, z);
    }

    __host__ __device__ inline void operator = (const Vec3& v) {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    __host__ __device__ inline void operator += (const Vec3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
    }

    __host__ __device__ inline void operator -= (const Vec3& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
    }

    __host__ __device__ inline void operator *= (float s) {
        x *= s;
        y *= s;
        z *= s;
    }

    __host__ __device__ inline void operator /= (float s) {
        x /= s;
        y /= s;
        z /= s;
    }
};

__host__ __device__ inline Vec3 operator + (const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__host__ __device__ inline Vec3 operator - (const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__host__ __device__ inline Vec3 operator * (const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

__host__ __device__ inline Vec3 operator * (const Vec3& v, float s) {
    return Vec3(v.x * s, v.y * s, v.z * s);
}

__host__ __device__ inline Vec3 operator / (const Vec3& v, float s) {
    return Vec3(v.x / s, v.y / s, v.z / s);
}