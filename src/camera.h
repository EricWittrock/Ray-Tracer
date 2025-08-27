#pragma once
#include "vec3.h"
#include "config.h"
#include "color.h"
#include "ray.h"

class Camera {
public:
    Camera();

    Vec3 position;
    Vec3 direction;

    void render(double pixels[IMAGE_WIDTH][IMAGE_WIDTH][3]) const;

private:
    Color castRay(const Ray& ray) const;
};