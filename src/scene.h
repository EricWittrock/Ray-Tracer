#pragma once
#include <vector>
#include "mesh.h"
#include "sphere.h"
#include "config.h"

class Scene {
public:
    Scene();
    ~Scene();

    std::vector<Mesh*> objects;
};