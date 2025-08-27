#pragma once
#include "vec3.h"


class Camera {
public:
    Camera();

    Vec3 position;
    Vec3 direction;

    void render() const;
};