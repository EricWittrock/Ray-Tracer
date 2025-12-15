#pragma once

#include "vec3.h"
#include "vector"
#include <fstream>
#include <string>
#include "material.h"
#include "BVHcreator.h"
#include "sceneConfigs.h"

namespace SceneAssets
{
    struct Texture {
        std::string name;
        std::string path;
        bool isLoaded = false;
        int offset = false;
        int width = 0;
        int height = 0;
    };
    struct Model {
        std::string name;
        std::string path;
    };
    struct Material {
        std::string name;
        char type;
        Vec3 color;
        float p1;
        float p2;
        float p3;
        std::vector<std::string> image_names;
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
        Vec3 rotation;
        float scale;
        bool importance_sample = false;
    };
    struct Sphere {
        std::string name;
        Vec3 position;
        float radius;
        std::string material_id;
    };
}

class SceneParser {
public:
    SceneConfigs scene_configs;
    std::vector<SceneAssets::Texture> textures;
    std::vector<SceneAssets::Model> models;
    std::vector<SceneAssets::Material> materials;
    std::vector<SceneAssets::Object> objects;
    std::vector<SceneAssets::ObjectInstance> object_instances;
    std::vector<SceneAssets::Sphere> spheres;
    std::string background_image_name = "";

    void parseFromFile(const char* path);

    const std::string nth_word(const std::string& line, int n);

    void parseLineImageDef(const std::string& line, const std::string& name);
    void parseLineModelDef(const std::string& line, const std::string& name);
    void parseLineMaterialDef(const std::string& line, const std::string& name);
    void parseLineObjectDef(const std::string& line, const std::string& name);
    void parseLineObjectInstance(const std::string& line, const std::string& name);
    void parseLineSphere(const std::string& line, const std::string& name);


    SceneAssets::Texture* getTextureByName(std::string name);
    SceneAssets::Model* getModelByName(std::string name);
    SceneAssets::Material* getMaterialByName(std::string name);
    SceneAssets::Object* getObjectByName(std::string name);
    SceneAssets::ObjectInstance* getObjectInstanceByName(std::string name);
    SceneAssets::Sphere* getSphereByName(std::string name);
    int getMaterialIndexByName(std::string name);

    void getTriangleData(float** tris, size_t* arr_len, BVH::BVHNode** bvh_nodes, int* num_bvh_nodes, int** is_tris, int* num_is_tris);
    void getMaterialData(const Material** out_materials, size_t* num_materials, const float** out_texture, size_t* texture_length);
    void getSceneConfigs(SceneConfigs* out_scene_configs);

private:
    int unique_string_counter = 0;
    std::string get_unique_string();
    void tryLoadTexture(Material& mat, int imageIndex, std::string imageName, std::vector<float>& pixels);
};
