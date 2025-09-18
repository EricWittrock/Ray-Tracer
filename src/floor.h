#pragma once
#include "object.h"

class Floor : public Object {
public:
    __device__ Floor(float elevation) : elevation(elevation) {}

    __device__ bool intersection(const Ray& ray, Vec3& pos, Vec3& normal) const 
    {
        if (ray.direction.y >= 0) {
            return false;
        }

        float t = (elevation - ray.position.y) / ray.direction.y;
        if (t < 0) return false;

        pos = ray.position + ray.direction * t;
        normal = Vec3(0, 1, 0);

        return true;
    }

private:
    float elevation;
};