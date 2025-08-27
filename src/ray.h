#pragma once
#include "vec3.h"

class Ray {
public:
    Ray(const Vec3& origin, const Vec3& direction)
        : position(origin), direction(direction) {}
    Vec3 position;
    Vec3 direction;
};