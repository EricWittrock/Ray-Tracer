#pragma once
#include "vec3.h"
#include "config.h"
#include "color.h"
#include "ray.h"
#include "scene.h"

class Camera {
public:
    Camera();

    Vec3 position;
    Vec3 direction;

    void setScene(Scene* s);
    void render(double pixels[IMAGE_WIDTH][IMAGE_WIDTH][3]) const;

private:
    Color castRay(const Ray& ray) const;
    Scene* scene;
};