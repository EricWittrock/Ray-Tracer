#pragma once
#include <cmath>

class Vec3 {
public:
    double x, y, z;
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

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

    Vec3 normalize() const {
        double invlen = 1.0 / length();
        if (invlen > 0) {
            return Vec3(x * invlen, y * invlen, z * invlen);
        }
        return Vec3(0, 0, 0);
    }

    void normalizeSelf() {
        double invlen = 1.0 / length();
        if (invlen > 0) {
            x *= invlen;
            y *= invlen;
            z *= invlen;
        }
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

inline Vec3 operator * (const Vec3& v, double s) {
    return Vec3(v.x * s, v.y * s, v.z * s);
}

inline Vec3 operator / (const Vec3& v, double s) {
    return Vec3(v.x / s, v.y / s, v.z / s);
}