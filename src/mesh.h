#pragma once

#include "vec3.h"
#include "ray.h"

class Mesh {
public:
    virtual ~Mesh() = default;
    virtual bool intersection(const Ray& ray, Vec3& pos, Vec3& normal) const = 0;
};
