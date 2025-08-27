#pragma once
#include "mesh.h"

class Sphere : public Mesh {
public:
    Sphere(const Vec3& center, double radius);

    Vec3 intersection(const Ray& ray) const override;

private:
    Vec3 center;
    double radius;
};