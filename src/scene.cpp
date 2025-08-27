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

void Scene::render(double pixels[IMAGE_WIDTH][IMAGE_WIDTH][3]) const {
    // camera.render();

    for (int j = 0; j < IMAGE_WIDTH; j++) {
        for (int i = 0; i < IMAGE_WIDTH; i++) {
            auto r = double(i) / (IMAGE_WIDTH-1);
            auto g = double(j) / (IMAGE_WIDTH-1);
            auto b = 0.0;

            pixels[i][j][0] = r;
            pixels[i][j][1] = g;
            pixels[i][j][2] = b;
        }
    }
}
