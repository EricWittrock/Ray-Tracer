#pragma once

#include "material.h"
#include "object.h"

class Sphere : public Object {
public:
    __device__ Sphere(const Vec3& center, double radius) : center(center), radius(radius) {}

    __device__ bool intersection(const Ray& ray, Vec3& pos, Vec3& normal) const 
    {
        Vec3 fromSphere = ray.position - center;
        // a = 1
        float b = 2.0f * fromSphere.dot(ray.direction);
        if(b > 0) { // don't hit from backwards
            return false;
        }
        float c = fromSphere.lengthSqr() - radius * radius;
        float d = b * b - 4.0f * c;
        if (d < 0) {
            return false;
        }
        float t = (-b - sqrt(d)) / 2.0f;
        pos = ray.position + ray.direction * t;
        normal = (pos - center).normalize();
        return true;
    }

private:
    Vec3 center;
    float radius;
    Material material;
};