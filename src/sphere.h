#pragma once

#include "object.h"
#include "material.h"
#include <curand_kernel.h>

class Sphere : public Object {
public:
    __host__ __device__ Sphere(const Vec3& center, float radius) : center(center), radius(radius) {}

    __device__ bool intersection(const Ray& ray, Vec3& out_pos, Vec3& out_normal) const // vestigial
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
        out_pos = ray.position + ray.direction * t;
        out_normal = (out_pos - center).normalize();
        return true;
    }

    __device__ static bool intersectSphere(const Ray& ray, Vec3 spherePos, float sphereRadius, Vec3& out_pos, Vec3& out_normal) {
        Vec3 fromSphere = ray.position - spherePos;
        // a = 1
        float b = 2.0f * fromSphere.dot(ray.direction);
        if(b > 0) { // don't hit from backwards
            return false;
        }
        float c = fromSphere.lengthSqr() - sphereRadius * sphereRadius;
        float d = b * b - 4.0f * c;
        if (d < 0) {
            return false;
        }
        float t = (-b - sqrt(d)) / 2.0f;
        out_pos = ray.position + ray.direction * t;
        out_normal = (out_pos - spherePos).normalize();
        return true;
    }

    __device__ static Vec3 randSphereVec(curandState* randState) 
    {
        float r1 = curand_uniform(randState);
        float r2 = curand_uniform(randState);
        float theta = r1 * 2.0f * 3.14159f;
        float phi = acosf(2.0f * r2 - 1.0f);
        float x = sinf(phi) * cosf(theta);
        float y = sinf(phi) * sinf(theta);
        float z = cosf(phi);
        return Vec3(x, y, z);
    }

    __device__ static Vec3 getTextureColor(float u, float v, const float* textures, int image_id, const Material& mat) {
        float px = u * mat.image_width;
        float py = v * mat.image_height;
        int offset = image_id == 0 ? mat.image1_offset : (image_id == 1 ? mat.image2_offset : mat.image3_offset);
        if (offset < 0) return Vec3(1.0f, 0.0f, 1.0f); // the color of error

        int i = offset + 3 * (int)px + 3 * mat.image_width * (int)py;
        return Vec3(
            textures[i],
            textures[i + 1],
            textures[i + 2]
        );
    }

    // spheres have limited material support
    // you can load a sphere as a mesh instead for full material options
    // This is just to meet the requirement of sphere intesections and textured spheres
    __device__ static bool hitSphere(Ray& ray, const Vec3& normal, const Vec3& hitPos, const Material& mat, const float* textures, curandState* randState) {
        float u = atan2(ray.direction.z, ray.direction.x) / (2.0f * 3.14159f) + 0.5f;
        float v = -asin(ray.direction.y) / (2.0f * 3.14159f) + 0.5f;

        Vec3 clr = getTextureColor(u, v, textures, 0, mat);
        ray.diffuseMultiplier = ray.diffuseMultiplier * clr;
        ray.position = hitPos;
        Vec3 randomDir = randSphereVec(randState);
        ray.direction = (normal + randomDir).normalize();

        ray.marchForward(1e-5f);
    }


private:
    Vec3 center;
    float radius;
};