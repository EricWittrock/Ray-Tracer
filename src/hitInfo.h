#pragma once
#include "vec3.h"

class HitInfo {
public:
    float t;
    Vec3 normal;
    Vec3 point;
    float u;
    float v;

    HitInfo(float t, const Vec3& normal, const Vec3& point) :
        t(t),
        normal(normal),
        point(point),
        u(u),
        v(v) {}
};