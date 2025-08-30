#pragma once
#include <vector>
#include "mesh.h"
#include "sphere.h"
#include "config.h"
#include "texture.h"

class Scene {
public:
    Scene();
    ~Scene();

    std::vector<Mesh*> objects;
    Texture environmentMap = Texture("C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\textures\\cape_hill_1k.hdr");
};