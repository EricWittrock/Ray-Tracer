#pragma once
#include "mesh.h"

class Sphere : public Mesh {
public:
    Sphere(const Vec3& center, double radius);

    bool intersection(const Ray& ray, Vec3& pos, Vec3& normal) const override;

private:
    Vec3 center;
    double radius;
};