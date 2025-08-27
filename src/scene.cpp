#include "scene.h"
#include "config.h"


Scene::Scene() {
    objects.push_back(new Sphere(Vec3(0, 0, -5), 1));
}

Scene::~Scene() {
    for (Mesh* obj : objects) {
        delete obj;
    }
}