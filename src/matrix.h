#pragma once
#include "vec3.h"

class Matrix {
public:
    float data[9];

    __host__ __device__ Matrix() {
        for (int i = 0; i < 9; i++) {
            data[i] = 0.0f;
        }
    }

    __host__ __device__ static Matrix rotationMatrix(Vec3& v) {
        Matrix mat;

        float cosX = cos(v.x);
        float sinX = sin(v.x);
        float cosY = cos(v.y);
        float sinY = sin(v.y);
        float cosZ = cos(v.z);
        float sinZ = sin(v.z);

        mat.data[0] = cosY * cosZ;
        mat.data[1] = -cosY * sinZ;
        mat.data[2] = sinY;

        mat.data[3] = cosX * sinZ + sinX * sinY * cosZ;
        mat.data[4] = cosX * cosZ - sinX * sinY * sinZ;
        mat.data[5] = -sinX * cosY;

        mat.data[6] = sinX * sinZ - cosX * sinY * cosZ;
        mat.data[7] = sinX * cosZ + cosX * sinY * sinZ;
        mat.data[8] = cosX * cosY;

        return mat;
    }

    __host__ __device__ static Matrix scaleMatrix(Vec3& v) {
        Matrix mat;
        mat.data[0] = 1.0f * v.x;
        mat.data[4] = 1.0f * v.y;
        mat.data[8] = 1.0f * v.z;
        return mat;
    }
};

__host__ __device__ inline Vec3 operator * (const Matrix& m, const Vec3& v) {
    Vec3 result;
    result.x = m.data[0] * v.x + m.data[1] * v.y + m.data[2] * v.z;
    result.y = m.data[3] * v.x + m.data[4] * v.y + m.data[5] * v.z;
    result.z = m.data[6] * v.x + m.data[7] * v.y + m.data[8] * v.z;
    return result;
}