#pragma once

#include "vec3.h"
#include "ray.h"

class Mesh {
public:
    virtual Vec3 intersection(const Ray& ray) const = 0;
};
