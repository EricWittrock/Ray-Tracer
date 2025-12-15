#pragma once

#define MAX_BOUNCES 6

#define ENABLE_SKYBOX true
#define BACKGROUND_BRIGHTNESS 0.5f
#define BACKGROUND_COLOR Vec3(1.0f, 1.0f, 0.05f)

#define ENABLE_VOLUME_SCATTERING true
#define ENABLE_DEFOCUS_BLUR true
#define ENABLE_ANTIALIASING true
#define HDRI_EXPORT false
#define ENABLE_SPHERES false
#define ENABLE_BVH true
#define BVH_DEPTH 16
#define BVH_NODE_MIN_TRIS 4

#define SCENE_CONFIG_PATH "C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\scene.txt"
#define OUTPUT_IMAGE_PATH "C:\\Users\\ericj\\Desktop\\HW\\CS336\\Ray-Tracer\\output\\output"