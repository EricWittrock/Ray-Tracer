#pragma once
#include <cmath>
// #include <cstdlib>
// #include <random>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif


class Vec3 {
public:
    float x, y, z;
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3() : x(0.0f), y(0.0f), z(0.0f) {}


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

    __host__ __device__ inline void reflect(const Vec3& normal) {
        float d = 2.0f * dot(normal);
        x = x - d * normal.x;
        y = y - d * normal.y;
        z = z - d * normal.z;
    }

    __host__ __device__ inline void refract(const Vec3& normal, float etai_over_etat) {
        float cos_theta = fminf(-this->dot(normal), 1.0f);
        Vec3 r_out_perp = Vec3(
            (x + normal.x * cos_theta) * etai_over_etat,
            (y + normal.y * cos_theta) * etai_over_etat,
            (z + normal.z * cos_theta) * etai_over_etat);
        Vec3 r_out_parallel = Vec3(
            normal.x * -std::sqrtf(fabsf(1.0f - r_out_perp.lengthSqr())),
            normal.y * -std::sqrtf(fabsf(1.0f - r_out_perp.lengthSqr())),
            normal.z * -std::sqrtf(fabsf(1.0f - r_out_perp.lengthSqr())));
        x = r_out_perp.x + r_out_parallel.x;
        y = r_out_perp.y + r_out_parallel.y;
        z = r_out_perp.z + r_out_parallel.z;
    }

    __host__ __device__ inline void rotate(const Vec3& r) {
        float temp1;
        float temp2;
        float cx = std::cos(r.x);
        float sx = std::sin(r.x);
        temp1 =  cx * y - sx * z;
        temp2 =  sx * y + cx * z;
        y = temp1; 
        z = temp2;

        float cy = std::cos(r.y);
        float sy = std::sin(r.y);
        temp1 =  cy * x + sy * z;
        temp2 = -sy * x + cy * z;
        x = temp1; 
        z = temp2;

        float cz = std::cos(r.z);
        float sz = std::sin(r.z);
        temp1 =  cz * x - sz * y;
        temp2 =  sz * x + cz * y;
        x = temp1; 
        y = temp2;
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

    __host__ __device__ int colorToInt() const {
        int r = static_cast<int>(fminf(fmaxf(x * 255.0f, 0.0f), 255.0f));
        int g = static_cast<int>(fminf(fmaxf(y * 255.0f, 0.0f), 255.0f));
        int b = static_cast<int>(fminf(fmaxf(z * 255.0f, 0.0f), 255.0f));
        return (r << 16) | (g << 8) | b;
    }

    __host__ __device__ static Vec3 fromColorInt(int color) {
        float r = static_cast<float>((color >> 16) & 0xFF) / 255.0f;
        float g = static_cast<float>((color >> 8) & 0xFF) / 255.0f;
        float b = static_cast<float>(color & 0xFF) / 255.0f;
        return Vec3(r, g, b);
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

__host__ __device__ inline Vec3 operator + (const Vec3& v, float s) {
    return Vec3(v.x + s, v.y + s, v.z + s);
}

__host__ __device__ inline Vec3 operator * (const Vec3& v, float s) {
    return Vec3(v.x * s, v.y * s, v.z * s);
}

__host__ __device__ inline Vec3 operator / (const Vec3& v, float s) {
    return Vec3(v.x / s, v.y / s, v.z / s);
}