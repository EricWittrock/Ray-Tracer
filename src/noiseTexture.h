#pragma once

#include "vec3.h"

class NoiseTexture {
public:
    int width;
    int height;
    float* data;

    NoiseTexture(int w, int h);

    ~NoiseTexture();

    float fourier_series(const Vec3& uv);
    void generate();
};