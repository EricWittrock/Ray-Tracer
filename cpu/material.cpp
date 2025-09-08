#include "material.h"


Vec3 Material::reflect(Ray& ray, const Vec3& normal, const Vec3& pos) const {

    Vec3 newDir = ray.direction.reflect((normal + Vec3::random() * roughness).normalize());
    ray.position = pos;
    ray.direction = newDir;
    ray.marchForward(0.0001);

    return color;
}