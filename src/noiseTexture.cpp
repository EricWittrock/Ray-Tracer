#include "noiseTexture.h"

NoiseTexture::NoiseTexture(int width, int height) {
    this->width = width;
    this->height = height;
    data = new float[width * height * 3];
}

NoiseTexture::~NoiseTexture() {
    delete[] data;
}

float NoiseTexture::fourier_series(const Vec3& uv2) {
    float h = 0.0f;
	h += std::sinf(uv2.x * 1.52f + uv2.y * 23.0f);
    h += std::sinf(uv2.x * 9.5f + uv2.y * 10.0f) * 0.9f;
    h += std::sinf(uv2.x * 15.5f + uv2.y * 23.0f) * 0.7f;
    h += std::sinf(uv2.x * 35.0f + uv2.y * 2.0f) * 0.5f;
    return h;
}

void NoiseTexture::generate() {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float u = static_cast<float>(x) / width;
            float v = static_cast<float>(y) / height;
            Vec3 uv2(u, v, 0.0);
            uv2.x += fourier_series(uv2 + Vec3(0.2, 0.9, 0.0)) * 0.1;
            uv2.y += fourier_series(uv2 + Vec3(0.1, 0.3, 0.0)) * 0.1;
            float h = fourier_series(uv2);
            int index = (y * width + x) * 3;

            // normalize with sigmoid function
            h = 1.0 / (1.0 + std::exp(-h * 2.0));

            data[index] = h;
            data[index + 1] = h;
            data[index + 2] = h;
        }
    }
}