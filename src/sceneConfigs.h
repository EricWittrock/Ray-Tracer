#pragma once
#include "vec3.h"

class SceneConfigs {
public:
    int outputWidth = 512;
    int outputHeight = 512;
    int numSamples = 1000;
    float focalLength = 1.0f;
    float apertureSize = 0.0f;
    float focusDistance = 4.0f;
    Vec3 cameraPos = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 cameraRot = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 cameraVel = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 cameraAngularVel = Vec3(0.0f, 0.0f, 0.0f);
    int envTextureWidth = 4096;
    int envTextureHeight = 2048;
};
