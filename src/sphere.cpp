#include "sphere.h"
#include "vec3.h"
#include "ray.h"

Sphere::Sphere(const Vec3& center, double radius) : center(center), radius(radius) {}

bool Sphere::intersection(const Ray& ray, Vec3& pos) const {
    Vec3 oc = ray.position - center;
    double a = ray.direction.lengthSqr();
    double b = 2.0 * oc.dot(ray.direction);
    double c = oc.lengthSqr() - radius * radius;
    double d = b * b - 4 * a * c;
    if (d < 0) {
        return false;
    }
    double t = (-b - sqrt(d)) / (2.0 * a);
    pos = ray.position + ray.direction * t;
    return true;
}