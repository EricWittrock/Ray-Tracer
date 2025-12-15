#pragma once

#include "vec3.h"
#include "ray.h"
#include <curand_kernel.h>
#include "rayAttractor.h"


class Material
{
public:
    char type;
    int color;
    float p1;
    float p2;
    float p3;
    int image1_offset;
    int image2_offset;
    int image3_offset;
    unsigned short image_width;
    unsigned short image_height;
    float* textures;

    __host__ __device__ Material()
        : type(0),
          color(0xFFFFFF),
          p1(0.0f),
          p2(0.0f),
          p3(0.0f),
          image1_offset(-1),
          image2_offset(-1),
          image3_offset(-1),
          image_width(0),
          image_height(0),
          textures(nullptr)
    {
    }

    // return true if terminated
    // hitMaterial.reflect(ray, hitTriIndex, tris, textures, &randState);
    __device__ bool reflect(Ray &ray, const Vec3 &hitPos, int hitTriIndex, float* tris, float* textures, curandState* randState) const
    // __device__ bool reflect(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, const Material &material, curandState* randState) const
    {
        Vec3 v0(tris[hitTriIndex + 0], tris[hitTriIndex + 1], tris[hitTriIndex + 2]);
        Vec3 v1(tris[hitTriIndex + 3], tris[hitTriIndex + 4], tris[hitTriIndex + 5]);
        Vec3 v2(tris[hitTriIndex + 6], tris[hitTriIndex + 7], tris[hitTriIndex + 8]);

        Vec3 n0(tris[hitTriIndex + 9], tris[hitTriIndex + 10], tris[hitTriIndex + 11]);
        Vec3 n1(tris[hitTriIndex + 12], tris[hitTriIndex + 13], tris[hitTriIndex + 14]);
        Vec3 n2(tris[hitTriIndex + 15], tris[hitTriIndex + 16], tris[hitTriIndex + 17]);
        Vec3 bary = barycentric(v0, v1, v2, hitPos);
        Vec3 normal = interpolateNormal(n0, n1, n2, bary);


        float tu1 = tris[hitTriIndex + 18];
        float tv1 = tris[hitTriIndex + 19];
        float tu2 = tris[hitTriIndex + 20];
        float tv2 = tris[hitTriIndex + 21];
        float tu3 = tris[hitTriIndex + 22];
        float tv3 = tris[hitTriIndex + 23];
        float u = bary.x * tu1 + bary.y * tu2 + bary.z * tu3;
        float v = bary.x * tv1 + bary.y * tv2 + bary.z * tv3;
        Vec3 clr = Vec3::fromColorInt(color);
        Vec3 clr0 = clr;
        Vec3 clr1 = clr;
        Vec3 clr2 = clr;
        if (image1_offset >= 0) {
            clr0 = getTextureColor(u, v, textures, 0);
        }
        if (image2_offset >= 0) {
            clr1 = getTextureColor(u, v, textures, 1);
        }
        if (image3_offset >= 0) {
            clr2 = getTextureColor(u, v, textures, 2);
        }

        // some materials don't importance sample because they are either legacy or perfect mirrors
        switch (type) {
            case 0:
                reflectType0(ray, normal, hitPos, randState);
                break;
            case 1:
                reflectType1(ray, normal, hitPos, randState);
                return true; // terminate after emissive hit
            case 2:
                return reflectType2(ray, normal, hitPos, randState);
                break;
            case 3:
                reflectType3(ray, normal, hitPos, clr0, randState);
                break;
            case 4:
                reflectType4(ray, normal, hitPos, randState);
                break;
            case 5:
                reflectType5(ray, normal, hitPos, randState);
                break;
            case 6:
                reflectType6(ray, normal, hitPos, randState);
                break;
            default:
                ray.emission = Vec3(1.0f, 0.0f, 1.0f); // the color of error
                return true;  
        }
        
        return false;
    };

private:

    __device__ Vec3 barycentric(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& pos) const {
        Vec3 v0 = b - a;
        Vec3 v1 = c - a;
        Vec3 v2 = pos - a;
        float d00 = v0.dot(v0);
        float d01 = v0.dot(v1);
        float d11 = v1.dot(v1);
        float d20 = v2.dot(v0);
        float d21 = v2.dot(v1);
        float denom = d00 * d11 - d01 * d01;
        float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom;
        float u = 1.0f - v - w;
        return Vec3(u, v, w);
    }

    __device__ Vec3 interpolateNormal(const Vec3& n0, const Vec3& n1, const Vec3& n2, const Vec3& bary) const {
        return (n0 * bary.x + n1 * bary.y + n2 * bary.z).normalize();
    }

    __device__ Vec3 randSphereVec(curandState* randState) const 
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

    __device__ Vec3 randHemisphereVec(const Vec3 &normal, curandState* randState) const 
    {
        Vec3 inUnitSphere = randSphereVec(randState);
        if (inUnitSphere.dot(normal) > 0.0f) {
            return inUnitSphere;
        } else {
            return inUnitSphere * -1.0f;
        }
    }

    __device__ Vec3 randCosineHemisphere(const Vec3 &normal, curandState* randState) const 
    {
        Vec3 inUnitSphere = randSphereVec(randState);
        float cos_theta = inUnitSphere.dot(normal);
        if (cos_theta > 0.0f) {
            return inUnitSphere * sqrtf(cos_theta);
        } else {
            return inUnitSphere * -1.0f * sqrtf(-cos_theta);
        }
    }

    __device__ Vec3 getTextureColor(float u, float v, float* textures, int image_id) const {
        float px = u * image_width;
        float py = v * image_height;
        int offset = image_id == 0 ? image1_offset : (image_id == 1 ? image2_offset : image3_offset);
        if (offset < 0) return Vec3(1.0f, 0.0f, 1.0f); // the color of error

        int i = offset + 3 * (int)px + 3 * image_width * (int)py;
        return Vec3(
            textures[i],
            textures[i + 1],
            textures[i + 2]
        );
    }

    __device__ float lambertian_scatter_pdf(const Vec3& dir_in, const Vec3& dir_out, const Vec3& normal) const {
        float cos_theta = normal.dot(dir_out);
        return cos_theta < 0 ? 0 : cos_theta / 3.14159f;
    }

    __device__ void reflectType0(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, curandState* randState) const 
    {
        Vec3 clr = Vec3::fromColorInt(color);
        ray.position = hitPos;
        
        if(curand_uniform(randState) < p1) { // clear coat reflection
            ray.diffuseMultiplier = ray.diffuseMultiplier * Vec3(0.95f, 0.95f, 0.95f);
            ray.direction.reflect(normal);
        }
        else { // diffuse reflection
            ray.diffuseMultiplier = ray.diffuseMultiplier * clr;
            Vec3 randVec = Vec3(
                curand_normal(randState),
                curand_normal(randState),
                curand_normal(randState)
            );
            ray.direction.reflect((normal + randVec * p2).normalize());
        }
        
        ray.marchForward(1e-5f);
    }

    // emissive
    __device__ void reflectType1(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, curandState* randState) const 
    {
        if (ray.direction.dot(normal) > 0.0f) {
            ray.emission = Vec3(0.0f, 0.0f, 0.0f);
            return;
        }
        Vec3 clr = Vec3::fromColorInt(color);
        ray.emission = clr * p1; // p1 = emission strength
    }

    // legacy dielectric (no pdf)
    __device__ bool reflectType2(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, curandState* randState) const 
    {
        ray.position = hitPos;
        float dotNorm = ray.direction.dot(normal); // if < 0: hits from the outside
        bool fromOutside = (dotNorm < 0.0f);
        float oldRI = ray.refractiveIndex;
        float newRI = fromOutside ? oldRI + p1 : oldRI - p1; // p1 = change in refractive index
        Vec3 normal2 = fromOutside ? normal : normal * -1.0f;

        float ri = oldRI / newRI; // equivalent to float ri = fromOutside ? (1.0f/1.5f) : (1.5f/1.0f);
        // float ri = fromOutside ? (1.0f/1.5f) : (1.5f/1.0f);
        float cos_theta = std::fminf(-ray.direction.dot(normal2), 1.0f);
        float sin_theta = std::sqrt(1.0f - cos_theta*cos_theta);
        bool will_refract = ri * sin_theta <= 1.0f;

        if (will_refract) {
            float r0 = (1 - ri) / (1 + ri);
            r0 = r0 * r0;
            r0 = r0 + (1-r0) * std::pow((1 - cos_theta), 5);
            if (r0 > curand_uniform(randState)) will_refract = false;
        }

        if (will_refract) { // refract
            ray.direction.refract(normal2, ri);
            ray.refractiveIndex = newRI;
        }
        else // reflect
        { 
            Vec3 randVec = Vec3(
                curand_normal(randState),
                curand_normal(randState),
                curand_normal(randState)
            );
            ray.direction.reflect(normal2);
        }

        ray.marchForward(1e-5f);
        return false;
    }

    // lambertian
    __device__ void reflectType3(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, const Vec3 &clr, curandState* randState) const 
    {
        Vec3 normal2 = (normal.dot(ray.direction) < 0.0f) ? normal : (normal * -1.0f);
        
        ray.diffuseMultiplier = ray.diffuseMultiplier * clr;
        ray.position = hitPos;
        Vec3 randomDir = randSphereVec(randState);
        ray.direction = (normal2 + randomDir).normalize();
        ray.marchForward(1e-5f);
    }

    // metal
    __device__ void reflectType4(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, curandState* randState) const 
    {
        Vec3 clr = Vec3::fromColorInt(color);
        ray.diffuseMultiplier = ray.diffuseMultiplier * clr;
        ray.position = hitPos;
        ray.direction.reflect(normal);
        ray.direction += randSphereVec(randState) * p1;
        ray.marchForward(1e-5f);
    }

    // lambertian pdf
    __device__ void reflectType5(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, curandState* randState) const 
    {
        Vec3 normal2 = (normal.dot(ray.direction) < 0.0f) ? normal : (normal * -1.0f);
        Vec3 incoming_direction = ray.direction; // save incoming direction
        
        ray.position = hitPos;

        bool importanceSample = true;
        float pdf_value;
        Vec3 outgoing_direction;
        
        if (importanceSample) {
            RayAttractor rayAttractor(Vec3(0.0f, 2.95f, -5.01f), Vec3(0.0f, 0.0f, 1.0f), Vec3(1.0f, 0.0f, 0.0f));
            if (curand_uniform(randState) < 0.5f) {
                outgoing_direction = rayAttractor.generate_random(hitPos, randState);
            } else {
                // outgoing_direction = randHemisphereVec(normal2, randState);
                outgoing_direction = randCosineHemisphere(normal2, randState);
            }
            float cos_theta = normal2.dot(outgoing_direction);
            if (cos_theta < 0.0f) cos_theta = 0.0f;
            float brdf = cos_theta / 3.14159f;
            pdf_value = rayAttractor.pdf_value(hitPos, outgoing_direction) * 0.5f + (brdf) * 0.5f;
        } else {
            outgoing_direction = randHemisphereVec(normal2, randState);
            // outgoing_direction = randCosineHemisphere(normal2, randState);
            pdf_value = 1.0f / (2 * 3.14159f);
        }
        float cos_theta = normal2.dot(outgoing_direction);
        if (cos_theta < 0.0f) cos_theta = 0.0f;
        float brdf = cos_theta / 3.14159f;

        // float scatter_pdf = lambertian_scatter_pdf(incoming_direction, outgoing_direction, normal2);
        
        Vec3 clr = Vec3::fromColorInt(color);
        ray.diffuseMultiplier = ray.diffuseMultiplier * clr * brdf / pdf_value;
        ray.direction = outgoing_direction;
        ray.marchForward(1e-5f);
    }

    // volume
    __device__ void reflectType6(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, curandState* randState) const 
    {
        bool entering = (normal.dot(ray.direction) < 0.0f);
        if (entering) {
            ray.volumeDensity += p1;
            Vec3 clr = Vec3::fromColorInt(color);
            ray.volumeColor = clr;
        }else {
            ray.volumeDensity -= p1;
        }

        ray.position = hitPos;
        ray.marchForward(1e-5f);
    }
};