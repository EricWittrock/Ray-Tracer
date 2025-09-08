#include "scene.h"
#include "config.h"


Scene::Scene() {
    Sphere* s = new Sphere(Vec3(2, -5, -15), 3.9);
    s->material.roughness = 0.0;
    s->material.color = Vec3(0.1, 1.0, 0.2);
    objects.push_back(s);

    s = new Sphere(Vec3(0, 3, -15), 3.8);
    s->material.roughness = 0.5;
    s->material.color = Vec3(0.1, 1.0, 0.2);
    objects.push_back(s);

    s = new Sphere(Vec3(-5, -6, -12), 2.3);
    s->material.roughness = 0.2;
    s->material.color = Vec3(0.1, 1.0, 0.2);
    objects.push_back(s);
}

Scene::~Scene() {
    for (Mesh* obj : objects) {
        delete obj;
    }
}