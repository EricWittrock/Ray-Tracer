#include "vec3.h"
#include "ray.h"

class Material {
public:
    Vec3 reflect(Ray&, const Vec3& normal, const Vec3& pos) const;

    Vec3 color = Vec3(1.0, 1.0, 1.0);
    double roughness = 0.0;
};