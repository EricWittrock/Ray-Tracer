#pragma once

#include "vec3.h"
#include "ray.h"

class Mesh {
public:
    virtual bool intersection(const Ray& ray, Vec3& pos) const = 0;
};
