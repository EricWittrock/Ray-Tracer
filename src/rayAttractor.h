#pragma once
#include "vec3.h"
#include <curand_kernel.h>

// used for importance sampling.
// RayAttractor is a quad
// It is automatically placed in the same position as a light quad
// rather than the light having a pdf itself (for flexibility)
class RayAttractor {
public:
    __device__ RayAttractor(const Vec3& pos, const Vec3& u, const Vec3& v) {
        this->pos = pos;
        this->u = u.normalize();
        this->v = v.normalize();
        normal = this->u.cross(this->v).normalize();
        area = u.length() * v.length();
    }

    // ray quad intersectionay 
    __device__ bool ray_intersect(const Vec3& origin, const Vec3& direction, Vec3 &intersection) const {
        float denom = normal.dot(direction);

        if (fabs(denom) < 1e-8) return false;

        float t = (pos - origin).dot(normal) / denom;
        if (t <= 0) return false;

        intersection = origin + direction * t;
        float u_pos = (intersection - pos).dot(this->u);
        float v_pos = (intersection - pos).dot(this->v);
        if (u_pos >= -0.5f && u_pos <= 0.5f && v_pos >= -0.5f && v_pos <= 0.5f) {
            return true;
        }
        return false;
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

    __device__ Vec3 generate_random(const Vec3& origin, curandState* randState) const {
        float r1 = curand_uniform(randState);
        float r2 = curand_uniform(randState);
        return pos + u * (r1 - 0.5f) + v * (r2 - 0.5f) - origin;
    }

private:
    Vec3 pos;
    Vec3 normal;
    Vec3 u;
    Vec3 v;
    float area;
};