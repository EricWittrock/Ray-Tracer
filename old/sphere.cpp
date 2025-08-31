#include "sphere.h"
#include "vec3.h"
#include "ray.h"

Sphere::Sphere(const Vec3& center, double radius) : center(center), radius(radius) {}

bool Sphere::intersection(const Ray& ray, Vec3& pos, Vec3& normal) const {
    Vec3 fromSphere = ray.position - center;
    // a = 1
    double b = 2.0 * fromSphere.dot(ray.direction);
    if(b > 0) { // don't hit from backwards
        return false;
    }
    double c = fromSphere.lengthSqr() - radius * radius;
    double d = b * b - 4 * c;
    if (d < 0) {
        return false;
    }
    double t = (-b - sqrt(d)) / 2.0;
    pos = ray.position + ray.direction * t;
    normal = (pos - center).normalize();
    return true;
}