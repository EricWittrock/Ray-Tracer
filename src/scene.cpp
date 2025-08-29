#include "scene.h"
#include "config.h"


Scene::Scene() {
    objects.push_back(new Sphere(Vec3(0, 0, -15), 4.9));
    objects.push_back(new Sphere(Vec3(0, 3, -15), 6.8));
}

Scene::~Scene() {
    // for (Mesh* obj : objects) {
    //     delete obj;
    // }
}