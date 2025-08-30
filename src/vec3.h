#pragma once
#include <cmath>
#include <cstdlib>
#include <random>

class Vec3 {
public:
    double x, y, z;
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    static Vec3 random() {
        return Vec3(
            (static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0,
            (static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0,
            (static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0
        );
    }

    static Vec3 randomGaussian(std::mt19937 &rng, double stddev = 1.0) {
        std::normal_distribution<double> dist(0.0, stddev);
        return Vec3(dist(rng), dist(rng), dist(rng));
    }

    inline double dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    inline Vec3 cross(const Vec3& v) const {
        return Vec3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }

    inline double lengthSqr() const {
        return x * x + y * y + z * z;
    }

    inline double length() const {
        return std::sqrt(lengthSqr());
    }

    Vec3 reflect(const Vec3& normal) const {
        double d = 2.0 * dot(normal);
        return Vec3(
            x - d * normal.x,
            y - d * normal.y,
            z - d * normal.z
        );
        // return this->copy();
    }

    Vec3 normalize() const {
        double invlen = 1.0 / length();
        if (invlen > 0) {
            return Vec3(x * invlen, y * invlen, z * invlen);
        }
        return Vec3(0, 0, 0);
    }

    static Vec3 normalize(Vec3& v) {
        double invlen = 1.0 / v.length();
        if (invlen > 0) {
            v.x *= invlen;
            v.y *= invlen;
            v.z *= invlen;
        }
        return v;
    }

    Vec3 copy() const {
        return Vec3(x, y, z);
    }

    inline void operator = (const Vec3& v) {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    inline void operator += (const Vec3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
    }

    inline void operator -= (const Vec3& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
    }

    inline void operator *= (double s) {
        x *= s;
        y *= s;
        z *= s;
    }

    inline void operator /= (double s) {
        x /= s;
        y /= s;
        z /= s;
    }
};

inline Vec3 operator + (const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

inline Vec3 operator - (const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

inline Vec3 operator * (const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

inline Vec3 operator * (const Vec3& v, double s) {
    return Vec3(v.x * s, v.y * s, v.z * s);
}

inline Vec3 operator / (const Vec3& v, double s) {
    return Vec3(v.x / s, v.y / s, v.z / s);
}