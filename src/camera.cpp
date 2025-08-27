#include "camera.h"
#include "mesh.h"
#include "scene.h"

Camera::Camera()
    : position(0, 0, 0),
      direction(0, 0, -1),
      scene(nullptr)
{
}

void Camera::setScene(Scene* s) {
    scene = s;
}

void Camera::render(double pixels[IMAGE_WIDTH][IMAGE_WIDTH][3]) const 
{
    for (int i = 0; i < IMAGE_WIDTH * IMAGE_WIDTH; i++) {
        int x = i % IMAGE_WIDTH;
        int y = i / IMAGE_WIDTH;
    }

    for (int j = 0; j < IMAGE_WIDTH; j++) {
        for (int i = 0; i < IMAGE_WIDTH; i++) {

            const double sx = (static_cast<double>(i) + 0.5) / IMAGE_WIDTH - 0.5;
            const double sy = (static_cast<double>(j) + 0.5) / IMAGE_WIDTH - 0.5;

            Vec3 dir = Vec3(sx, sy, -FOCAL_LENGTH).normalize(); // todo
            Ray ray(position, dir);

            Color clr = castRay(ray);

            pixels[i][j][0] = clr.r;
            pixels[i][j][1] = clr.g;
            pixels[i][j][2] = clr.b;
        }
    }
}

Color Camera::castRay(const Ray& ray) const {
    for (Mesh* obj : scene->objects) {
        Vec3 pos(0, 0, 0);
        bool intersects = obj->intersection(ray, pos);
        if(intersects) return Color(1.0, 0.0, 0.0);
    }

    return Color(0, 0.0, 0);
}