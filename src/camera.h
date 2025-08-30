#pragma once
#include "vec3.h"
#include "config.h"
#include "ray.h"
#include "scene.h"

class Camera {
public:
    Camera();

    Vec3 position;
    Vec3 right;
    Vec3 up;

    void setScene(Scene* s);
    void render(double pixels[IMAGE_WIDTH][IMAGE_WIDTH][3]);

private:
    Vec3 castRay(const Ray& ray) const;
    Scene* scene;
};