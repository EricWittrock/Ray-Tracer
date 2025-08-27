#pragma once
#include <vector>
#include "camera.h"
#include "mesh.h"
#include "sphere.h"
#include "config.h"

class Scene {
public:
    Scene();
    ~Scene();
    void render(double pixels[IMAGE_WIDTH][IMAGE_WIDTH][3]) const;

private:
    Camera camera;
    std::vector<Mesh*> objects;
};