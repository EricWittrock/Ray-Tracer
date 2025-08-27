#include "sphere.h"
#include "vec3.h"
#include "ray.h"

Sphere::Sphere(const Vec3& center, double radius) : center(center), radius(radius) {}

Vec3 Sphere::intersection(const Ray& ray) const {
    Vec3 oc = ray.position - center;
    double a = ray.direction.lengthSqr();
    double b = 2.0 * oc.dot(ray.direction);
    double c = oc.lengthSqr() - radius * radius;
    double d = b * b - 4 * a * c;
    if (d < 0) {
        return Vec3(0, 0, 0);  // No intersection
    }
    double t = (-b - sqrt(d)) / (2.0 * a);
    return ray.position + ray.direction * t;
}