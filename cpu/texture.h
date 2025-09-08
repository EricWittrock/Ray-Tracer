#pragma once
#include "../lib/stb_image/stb_image.h"
#include "../lib/stb_image/stb_image_write.h"
#include <string>
#include "vec3.h"
#include <cstring>


class Texture {
public:
    Texture(const char* path);
    Texture(const float* pixels, int width, int height);
    ~Texture();

    int width = 0;
    int height = 0;

    void save(std::string path) const;

    inline Vec3 getColor(int x, int y) const {
        if(x < 0 || x >= width || y < 0 || y >= height) {
            return Vec3(1.0, 0.0, 1.0);
        }
        int index = (x + y * width) * 3;
        return Vec3(data[index], data[index + 1], data[index + 2]);
    }

    inline const float* getData() const {
        return data;
    }

    inline unsigned int dataLength() const {
        return width * height * 3;
    }

    inline size_t sizeBytes() const {
        return width * height * 3 * sizeof(float);
    }

    static void saveImgData(const char* path, const float* data, int width, int height);
    static void saveImgData(const char* path, const unsigned char* data, int width, int height);


private:
    bool initFromFile = false;
    int bytesPerLine = 0;
    float* data = nullptr;
};