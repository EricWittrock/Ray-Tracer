#pragma once

#define IMAGE_WIDTH 512
#define FOCAL_LENGTH 0.3f
#define NUM_SAMPLES 500
#define MAX_BOUNCES 6

#define ENABLE_SKYBOX false
#define BACKGROUND_BRIGHTNESS 0.01f
#define BACKGROUND_COLOR Vec3(1.0f, 0.0f, 0.0f)

#define BVH_DEPTH 4
#define BVH_NODE_MIN_TRIS 2
#define SPECTRAL_RENDERING false

#define SCENE_CONFIG_PATH "C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\scene.txt"
#define OUTPUT_IMAGE_PATH "C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\output\\output.png"