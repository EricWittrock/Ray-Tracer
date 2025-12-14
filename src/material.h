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
          image_height(0)
    {
    }


    // return true if terminated
    __device__ bool reflect(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, const Material &material, curandState* randState) const
    {
        // some materials don't importance sample because they are either legacy or perfect mirrors
        switch (type) {
            case 0:
                reflectType0(ray, normal, hitPos, material, randState);
                break;
            case 1:
                reflectType1(ray, normal, hitPos, material, randState);
                return true; // terminate after emissive hit
            case 2:
                return reflectType2(ray, normal, hitPos, material, randState);
                break;
            case 3:
                reflectType3(ray, normal, hitPos, material, randState);
                break;
            case 4:
                reflectType4(ray, normal, hitPos, material, randState);
                break;
            case 5:
                reflectType5(ray, normal, hitPos, material, randState);
                break;
            default:
                ray.emission = Vec3(1.0f, 0.0f, 1.0f); // the color of error
                return true;  
        }
        
        return false;
    };

private:

    __device__ Vec3 randSphereVec(curandState* randState) const 
    {
        float r1 = curand_uniform(randState);
        float r2 = curand_uniform(randState);
        float theta = r1 * 2.0f * 3.14159265358979323846f;
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

    __device__ float lambertian_scatter_pdf(const Vec3& dir_in, const Vec3& dir_out, const Vec3& normal) const {
        float cos_theta = normal.dot(dir_out);
        return cos_theta < 0 ? 0 : cos_theta / 3.14159f;
    }

    __device__ void reflectType0(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, const Material &material, curandState* randState) const 
    {
        Vec3 color = Vec3::fromColorInt(material.color);
        ray.position = hitPos;
        
        if(curand_uniform(randState) < material.p1) { // clear coat reflection
            ray.diffuseMultiplier = ray.diffuseMultiplier * Vec3(0.95f, 0.95f, 0.95f);
            ray.direction.reflect(normal);
        }
        else { // diffuse reflection
            ray.diffuseMultiplier = ray.diffuseMultiplier * color;
            Vec3 randVec = Vec3(
                curand_normal(randState),
                curand_normal(randState),
                curand_normal(randState)
            );
            ray.direction.reflect((normal + randVec * material.p2).normalize());
        }
        
        ray.marchForward(1e-5f);
    }

    // emissive
    __device__ void reflectType1(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, const Material &material, curandState* randState) const 
    {
        if (ray.direction.dot(normal) > 0.0f) {
            ray.emission = Vec3(0.0f, 0.0f, 0.0f);
            return;
        }
        Vec3 color = Vec3::fromColorInt(material.color);
        ray.emission = color * material.p1; // p1 = emission strength
    }

    // dielectric
    __device__ bool reflectType2(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, const Material &material, curandState* randState) const 
    {
        ray.position = hitPos;
        float dotNorm = ray.direction.dot(normal); // if < 0: hits from the outside
        bool fromOutside = (dotNorm < 0.0f);
        float oldRI = ray.refractiveIndex;
        float newRI = (dotNorm < 0.0f) ? oldRI + p1 : oldRI - p1; // p1 = change in refractive index

        float ri = oldRI / newRI;
        float cos_theta = std::fminf(-ray.direction.dot(normal), 1.0f);
        float sin_theta = std::sqrt(1.0f - cos_theta*cos_theta);
        bool will_refract = ri * sin_theta <= 1.0f;

        if (will_refract) {
            float r0 = (1 - ri) / (1 + ri);
            r0 = r0 * r0;
            r0 = r0 + (1-r0) * std::pow((1 - cos_theta), 5);
            if (r0 > curand_uniform(randState)) will_refract = false;
        }

        if(!fromOutside) will_refract = true; // always refract when exiting

        if (will_refract) { // refract
            if (dotNorm > 0.0f) {
                ray.direction.refract(normal * -1, ri);
            } else {
                ray.direction.refract(normal, ri);
            }
        }
        else // reflect
        { 
            Vec3 randVec = Vec3(
                curand_normal(randState),
                curand_normal(randState),
                curand_normal(randState)
            );
            if (dotNorm > 0.0f) { // hit from inside
                ray.direction.reflect(normal * -1 + randVec * 0.02f);
                // ray.emission = Vec3(0.0f, 1.0f, 0.0f);
            } else { // hit from outside
                ray.direction.reflect(normal + randVec * 0.02f);
                // ray.emission = Vec3(1.0f, 0.0f, 1.0f);
            }
            // return true;
            
            // ray.direction.reflect((normal + randVec * 0.05f).normalize());
        }


        

        // ray.position = hitPos;
        // float eta = ray.refractiveIndex / p1; // p1 = refractive index of material
        // ray.direction.refract(normal, eta);
        // ray.refractiveIndex += p1;
        ray.marchForward(1e-5f);
        return false;
    }

    // lambertian
    __device__ void reflectType3(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, const Material &material, curandState* randState) const 
    {
        Vec3 normal2 = (normal.dot(ray.direction) < 0.0f) ? normal : (normal * -1.0f);
        
        Vec3 color = Vec3::fromColorInt(material.color);
        ray.diffuseMultiplier = ray.diffuseMultiplier * color;
        ray.position = hitPos;
        Vec3 randomDir = randSphereVec(randState);
        ray.direction = (normal2 + randomDir).normalize();
        ray.marchForward(1e-5f);
    }

    // metal
    __device__ void reflectType4(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, const Material &material, curandState* randState) const 
    {
        Vec3 color = Vec3::fromColorInt(material.color);
        ray.diffuseMultiplier = ray.diffuseMultiplier * color;
        ray.position = hitPos;
        ray.direction.reflect(normal);
        ray.direction += randSphereVec(randState) * material.p1;
        ray.marchForward(1e-5f);
    }

    // lambertian pdf
    __device__ void reflectType5(Ray &ray, const Vec3 &normal, const Vec3 &hitPos, const Material &material, curandState* randState) const 
    {
        Vec3 normal2 = (normal.dot(ray.direction) < 0.0f) ? normal : (normal * -1.0f);
        Vec3 incoming_direction = ray.direction; // save incoming direction
        
        ray.position = hitPos;

        bool importanceSample = false;
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
        
        Vec3 color = Vec3::fromColorInt(material.color);
        ray.diffuseMultiplier = ray.diffuseMultiplier * color * brdf / pdf_value;
        ray.direction = outgoing_direction;
        ray.marchForward(1e-5f);
    }
};