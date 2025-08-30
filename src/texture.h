#pragma once
#include "../lib/stb_image/stb_image.h"
#include <string>
#include "vec3.h"


class Texture {
public:
    Texture(std::string path);
    ~Texture();

    int width = 0;
    int height = 0;

    inline Vec3 getColor(int x, int y) const {
        if(x < 0 || x >= width || y < 0 || y >= height) {
            return Vec3(1.0, 0.0, 1.0);
        }
        int index = (x + y * width) * 3;
        return Vec3(data[index], data[index + 1], data[index + 2]);
    }

private:
    int bytesPerLine = 0;
    float* data = nullptr;
};