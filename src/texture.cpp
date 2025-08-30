#define STB_IMAGE_IMPLEMENTATION
#include "texture.h"
#include <iostream>


Texture::Texture(std::string path)
{
    int numChannels = 3;
    data = stbi_loadf(path.c_str(), &width, &height, &numChannels, 3);
    if (data == nullptr) {
        std::cerr << "Could not load texture: " << path << std::endl;
        return;
    }

    bytesPerLine = width * numChannels * sizeof(float);
}

Texture::~Texture() {
    delete[] data;
}
