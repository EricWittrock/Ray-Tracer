#pragma once
#include "vec3.h"
#include <curand_kernel.h>


// used for importance sampling.
class ImportanceTri {
public:
    Vec3 p1;
    Vec3 p2;
    Vec3 p3;
    float area;
    Vec3 normal;

    __device__ ImportanceTri() {
        area = -1.0f;
    }

    __device__ ImportanceTri(const Vec3& p1, const Vec3& p2, const Vec3& p3) {
        this->p1 = p1;
        this->p2 = p2;
        this->p3 = p3;
        this->normal = (p2 - p1).cross(p3 - p1).normalize();
        this->area = 0.5f * (p2 - p1).cross(p3 - p1).length();
    }

    __device__ void set_points(const Vec3& p1, const Vec3& p2, const Vec3& p3) {
        this->p1 = p1;
        this->p2 = p2;
        this->p3 = p3;
        this->normal = (p2 - p1).cross(p3 - p1).normalize();
        this->area = 0.5f * (p2 - p1).cross(p3 - p1).length();
    }

    // ray-tri intersection 
    __device__ bool ray_intersect(const Vec3& origin, const Vec3& direction, Vec3 &outPos) const {
        const float epsilon = 1e-9f;
        Vec3 edge1 = p2 - p1;
        Vec3 edge2 = p3 - p1;
        Vec3 ray_cross_e2 = direction.cross(edge2);

        float det = edge1.dot(ray_cross_e2);
        if (det > -epsilon && det < epsilon) { // ray is parallel to triangle plane
            return false; 
        }

        float inv_det = 1.0f / det;
        Vec3 s = origin - p1;
        float u = inv_det * s.dot(ray_cross_e2);
        if (u < 0.0f || u > 1.0f)
            return false;

        Vec3 q = s.cross(edge1);
        float v = inv_det * direction.dot(q);
        if (v < 0.0f || u + v > 1.0f)
            return false;

        float t = inv_det * edge2.dot(q);
        if (t > epsilon) {
            outPos = origin + direction * t;
            return true;
        } else {
            return false;
        }
    }

    __device__ float pdf_value(const Vec3& origin, const Vec3& direction) const {
        Vec3 intersection;
        bool intersected = ray_intersect(origin, direction, intersection);
        if (!intersected) return 0.0f;

        Vec3 to_intersection = intersection - origin;
        float distance_squared = to_intersection.lengthSqr();
        float cos_theta = fabsf(to_intersection.normalize().dot(normal));
        return distance_squared / (cos_theta * area);
    }

    // generate a random point on the triangle
    __device__ Vec3 generate_random(const Vec3& origin, curandState* randState) const {
        float r1 = curand_uniform(randState);
        float r2 = curand_uniform(randState);

        float su = sqrtf(r1);
        float b0 = 1.0f - su;
        float b1 = r2 * su;
        float b2 = 1.0f - b0 - b1;

        Vec3 random_point = p1 * b0 + p2 * b1 + p3 * b2;
        return (random_point - origin).normalize();
    }
};