#pragma once
#include "vec3.h"

class Ray {
public:
    Ray(const Vec3& origin, const Vec3& direction)
        : position(origin), direction(direction) {}

    void marchForward(double distance) {
        position += direction * distance;
    }

    Vec3 position;
    Vec3 direction;
    int numBounces = 0;
};