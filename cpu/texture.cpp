#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "texture.h"
#include <iostream>


Texture::Texture(const char* path)
{
    int numChannels = 3;
    data = stbi_loadf(path, &width, &height, &numChannels, 3);
    if (data == nullptr) {
        std::cerr << "Could not load texture: " << path << std::endl;
        return;
    }

    bytesPerLine = width * numChannels * sizeof(float);
    initFromFile = true;
}

Texture::Texture(const float* pixels, int width, int height)
    : width(width), height(height)
{
    bytesPerLine = width * 3 * sizeof(float);
    data = new float[width * height * 3];
    std::memcpy(data, pixels, width * height * 3 * sizeof(float));
}

Texture::~Texture() {
    if(data == nullptr) {
        std::cerr << "Texture data is nullptr in destructor" << std::endl;
        return;
    }
    if (initFromFile) {
        stbi_image_free(data);
    }else {
        delete[] data;
    }
}

void Texture::save(std::string path) const {
    unsigned char* img = new unsigned char[width * height * 3];
    for (int i = 0; i < width * height * 3; ++i) {
        img[i] = static_cast<unsigned char>(data[i] * 255);
    }
    stbi_write_png(path.c_str(), width, height, 3, img, width * 3);
    delete[] img;
}

void Texture::saveImgData(const char* path, const float* data, int width, int height) {
    unsigned char* img = new unsigned char[width * height * 3];
    for (int i = 0; i < width * height * 3; ++i) {
        img[i] = static_cast<unsigned char>(data[i] * 255);
    }
    stbi_write_png(path, width, height, 3, img, width * 3);
    delete[] img;
}

void Texture::saveImgData(const char* path, const unsigned char* data, int width, int height) {
    stbi_write_png(path, width, height, 3, data, width * 3);
}