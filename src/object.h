#pragma once

#include "vec3.h"
#include "ray.h"
#include "material.h"

class Object {
public:
    __device__ virtual bool intersection(const Ray& ray, Vec3& pos, Vec3& normal) const = 0;

    Material material;
};
