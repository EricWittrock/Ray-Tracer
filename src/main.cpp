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

            for (int c = 0; c < 3; c++) {
                if (pixels[i][j][c] < 0) pixels[i][j][c] = 0;
                if (pixels[i][j][c] > 1) pixels[i][j][c] = 1;
            }

            int r = static_cast<int>(255 * pixels[i][j][0]);
            int g = static_cast<int>(255 * pixels[i][j][1]);
            int b = static_cast<int>(255 * pixels[i][j][2]);

            std::cout << r << ' ' << g << ' ' << b << '\n';
        }
    }

    return 0;
}