#include <iostream>
#include "scene.h"
#include "config.h"
#include "camera.h"

double pixels[IMAGE_WIDTH][IMAGE_WIDTH][3];
Scene scene;
Camera camera;

int main() 
{
    camera.setScene(&scene);
    camera.render(pixels);

    std::cout << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_WIDTH << "\n255\n";
    for (int j = IMAGE_WIDTH-1; j >= 0; j--) {
        for (int i = 0; i < IMAGE_WIDTH; i++) {
            int ir = static_cast<int>(255.999 * pixels[i][j][0]);
            int ig = static_cast<int>(255.999 * pixels[i][j][1]);
            int ib = static_cast<int>(255.999 * pixels[i][j][2]);

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    return 0;
}