#include "camera.h"

Camera::Camera()
    : position(0, 0, 0),
      direction(0, 0, -1)
{
}


void Camera::render(double pixels[IMAGE_WIDTH][IMAGE_WIDTH][3]) const {
    for (int i = 0; i < IMAGE_WIDTH * IMAGE_WIDTH; i++) {
        int x = i % IMAGE_WIDTH;
        int y = i / IMAGE_WIDTH;
    }

    for (int j = 0; j < IMAGE_WIDTH; j++) {
        for (int i = 0; i < IMAGE_WIDTH; i++) {

            Vec3 dir = Vec3(double(i) / (IMAGE_WIDTH - 1), double(j) / (IMAGE_WIDTH - 1), -FOCAL_LENGTH).normalize(); // todo
            const Ray ray(position, dir);

            Color clr = castRay(ray);

            pixels[i][j][0] = clr.r;
            pixels[i][j][1] = clr.g;
            pixels[i][j][2] = clr.b;
        }
    }
}

Color Camera::castRay(const Ray& ray) const {
    return Color(0, 1.0, 0);
}