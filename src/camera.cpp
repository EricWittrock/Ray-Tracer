#include "camera.h"
#include "mesh.h"
#include "scene.h"

Camera::Camera()
    : position(0, 0, 0),
      right(1, 0, 0),
      up(0, 1, 0),
      scene(nullptr)
{
}

void Camera::setScene(Scene* s) {
    scene = s;
}

void Camera::render(double pixels[IMAGE_WIDTH][IMAGE_WIDTH][3]) 
{
    up.normalizeSelf();
    right.normalizeSelf();
    Vec3 forward = up.cross(right).normalize();

    for (int j = 0; j < IMAGE_WIDTH; j++) {
        for (int i = 0; i < IMAGE_WIDTH; i++) {

            const double sx = (static_cast<double>(i) + 0.5) / IMAGE_WIDTH - 0.5;
            const double sy = (static_cast<double>(j) + 0.5) / IMAGE_WIDTH - 0.5;

            Vec3 s = forward * FOCAL_LENGTH + right * sx + up * sy;

            Vec3 dir = (s - position).normalize();
            Ray ray(position, dir);

            Color clr = castRay(ray);

            pixels[i][j][0] = clr.r;
            pixels[i][j][1] = clr.g;
            pixels[i][j][2] = clr.b;
        }
    }
}

Color Camera::castRay(const Ray& ray) const {

    double nearestHitDistanceSqr = 1e10;
    Vec3 pos(0, 0, 0);
    Vec3 norm(0, 0, 0);
    for (Mesh* obj : scene->objects) {
        Vec3 newPos(0, 0, 0);
        Vec3 newNorm(0, 0, 0);
        bool intersects = obj->intersection(ray, newPos, newNorm);

        if(intersects && (newPos - position).lengthSqr() < nearestHitDistanceSqr) {
            nearestHitDistanceSqr = (newPos - position).lengthSqr();
            pos = newPos;
            norm = newNorm;
            // break;
        }
    }

    if(nearestHitDistanceSqr < 1e10) {
        Vec3 lightDir = Vec3(0, -1, 0.4).normalize();
        double diffuse = std::max(0.0, norm.dot(lightDir)) * 1.0;
        return Color(diffuse, diffuse, diffuse);
    }

    return Color(0, 0.0, 0);
}