#pragma once
#include "vec3.h"

class SceneConfigs {
public:
    int outputWidth = 512;
    int outputHeight = 512;
    int numSamples = 1000;
    float focalLength = 0.3f;
    Vec3 cameraPos = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 cameraRot = Vec3(0.0f, 0.0f, 0.0f);
    int envTextureWidth = 4096;
    int envTextureHeight = 2048;
};

/*

#pragma once

#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT IMAGE_WIDTH
#define FOCAL_LENGTH 0.3f
#define NUM_SAMPLES 1000
#define MAX_BOUNCES 6
#define SCENE_OFFSET Vec3(0.0f, 0.0f, 0.0f)

#define ENABLE_SKYBOX false
#define BACKGROUND_BRIGHTNESS 0.0f
#define BACKGROUND_COLOR Vec3(1.0f, 1.0f, 0.05f)

#define ENABLE_BVH true
#define BVH_DEPTH 16
#define BVH_NODE_MIN_TRIS 4

#define SCENE_CONFIG_PATH "C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\scene_cornellbox.txt"
#define OUTPUT_IMAGE_PATH "C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\output\\output.png"
*/