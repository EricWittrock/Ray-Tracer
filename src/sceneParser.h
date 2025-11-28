#pragma once

#include "vec3.h"
#include "vector"
#include <fstream>
#include <string>
#include "material.h"
#include "BVHcreator.h"

namespace SceneAssets
{
    struct Texture {
        std::string name;
        std::string path;
    };
    struct Model {
        std::string name;
        std::string path;
        std::string mtl_path;
    };
    struct Material {
        std::string name;
        char type;
        Vec3 albedo;
    };
    struct Object {
        std::string name;
        std::string model_name;
        std::string material_id;
    };
    struct ObjectInstance {
        std::string name;
        std::string object_name;
        Vec3 position;
        float scale;
    };
}

class SceneParser {
public:
    Vec3 camera_pos;
    Vec3 camera_look_at;
    float focal_length;
    int num_samples;
    int max_bounces;
    int image_width;
    int num_objects;
    int num_assets;
    std::string background_texture_name;
    std::vector<SceneAssets::Texture> textures;
    std::vector<SceneAssets::Model> models;
    std::vector<SceneAssets::Material> materials;
    std::vector<SceneAssets::Object> objects;
    std::vector<SceneAssets::ObjectInstance> object_instances;

    SceneParser::SceneParser()
    : camera_pos(0.0f, 0.0f, 0.0f),
      camera_look_at(0.0f, 0.0f, -1.0f),
      focal_length(0.3f),
      num_samples(1),
      max_bounces(1),
      image_width(512),
      num_objects(0),
      num_assets(0),
      background_texture_name("") {}

    void parseFromFile(const char* path);

    const std::string nth_word(const std::string& line, int n);

    void parseLineImageDef(const std::string& line, const std::string& name);
    void parseLineModelDef(const std::string& line, const std::string& name);
    void parseLineMaterialDef(const std::string& line, const std::string& name);
    void parseLineObjectDef(const std::string& line, const std::string& name);
    void parseLineObjectInstance(const std::string& line, const std::string& name);

    SceneAssets::Texture* getTextureByName(std::string name);
    SceneAssets::Model* getModelByName(std::string name);
    SceneAssets::Material* getMaterialByName(std::string name);
    SceneAssets::Object* getObjectByName(std::string name);
    SceneAssets::ObjectInstance* getObjectInstanceByName(std::string name);
    int getMaterialIndexByName(std::string name);

    void getTriangleData(float** tris, size_t* arr_len, BVH::BVHNode** bvh_nodes, int* num_bvh_nodes);
    void getMaterialData(const Material** materials, size_t* num_materials);
};
