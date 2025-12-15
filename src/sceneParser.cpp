#include "sceneParser.h"
#include "model.h"
#include "texture.h"
#include "noiseTexture.h"

namespace {
    enum Scope {
        GLOBAL,
        DEF_IMAGE,
        DEF_MODEL,
        DEF_MATERIAL,
        DEF_OBJECT,
        DEF_OBJECT_INSTANCE,
        DEF_SPHERE
    };
}

std::string SceneParser::get_unique_string() {
    return "rand_str_" + std::to_string(unique_string_counter++);
}

void SceneParser::parseFromFile(const char* path) {
    textures.clear();
    models.clear();
    materials.clear();
    objects.clear();
    object_instances.clear();

    std::ifstream file(path);
    std::string line;

    if (!file.is_open()) {
        std::cout << "Could not open config file." << std::endl;
        exit(1);
    }

    Scope scope = GLOBAL;
    std::string name;

    while (std::getline(file, line)) {
        if(line.empty()) {
            scope = GLOBAL;
            continue;
        }
        if (line.at(0) == '#') continue; // Ignore comments
        

        if (scope == DEF_IMAGE) {
            parseLineImageDef(line, name);
            continue;
        } else if (scope == DEF_MODEL) {
            parseLineModelDef(line, name);
            continue;
        } else if (scope == DEF_MATERIAL) {
            parseLineMaterialDef(line, name);
            continue;
        } else if (scope == DEF_OBJECT) {
            parseLineObjectDef(line, name);
            continue;
        } else if (scope == DEF_OBJECT_INSTANCE) {
            parseLineObjectInstance(line, name);
            continue;
        } else if (scope == DEF_SPHERE) {
            parseLineSphere(line, name);
            continue;
        }


        std::string first_word = nth_word(line, 0);

        if (first_word == "SET_OUTPUT_WIDTH") {
            scene_configs.outputWidth = std::stoi(nth_word(line, 1));
        } 
        else if (first_word == "SET_OUTPUT_HEIGHT") {
            scene_configs.outputHeight = std::stoi(nth_word(line, 1));
        }
        else if (first_word == "SET_NUM_SAMPLES") {
            scene_configs.numSamples = std::stoi(nth_word(line, 1));
        }
        else if (first_word == "SET_CAM_POS") {
            scene_configs.cameraPos = Vec3(
                std::stof(nth_word(line, 1)),
                std::stof(nth_word(line, 2)),
                std::stof(nth_word(line, 3))
            );
        } 
        else if (first_word == "SET_CAM_ROTATION") {
            scene_configs.cameraRot = Vec3(
                std::stof(nth_word(line, 1)),
                std::stof(nth_word(line, 2)),
                std::stof(nth_word(line, 3))
            );
        } 
        else if (first_word == "SET_CAM_VEL") {
            scene_configs.cameraVel = Vec3(
                std::stof(nth_word(line, 1)),
                std::stof(nth_word(line, 2)),
                std::stof(nth_word(line, 3))
            );
        }
        else if (first_word == "SET_CAM_ANGULAR_VEL") {
            scene_configs.cameraAngularVel = Vec3(
                std::stof(nth_word(line, 1)),
                std::stof(nth_word(line, 2)),
                std::stof(nth_word(line, 3))
            );
        }
        else if (first_word == "SET_FOCAL_LENGTH") {
            scene_configs.focalLength = std::stof(nth_word(line, 1));
        }
        else if (first_word == "SET_APERTURE_RADIUS") {
            scene_configs.apertureSize = std::stof(nth_word(line, 1));
        }
        else if (first_word == "SET_FOCUS_DISTANCE") {
            scene_configs.focusDistance = std::stof(nth_word(line, 1));
        }
        else if (first_word == "SET_BACKGROUND") {
            background_image_name = nth_word(line, 1);
        } 
        else if (first_word == "ADD_OBJECT") {
            scope = DEF_OBJECT_INSTANCE;
            name = get_unique_string();
            SceneAssets::ObjectInstance object_instance;
            object_instance.name = name;
            object_instances.push_back(object_instance);
        } 
        else if (first_word == "DEF_IMAGE") {
            scope = DEF_IMAGE;
            name = nth_word(line, 1);
            SceneAssets::Texture texture;
            texture.name = name;
            textures.push_back(texture);
        } 
        else if (first_word == "DEF_MODEL") {
            scope = DEF_MODEL;
            name = nth_word(line, 1);
            SceneAssets::Model model;
            model.name = name;
            models.push_back(model);
        } 
        else if (first_word == "DEF_MATERIAL") {
            scope = DEF_MATERIAL;
            name = nth_word(line, 1);
            SceneAssets::Material material;
            material.name = name;
            materials.push_back(material);
        } 
        else if (first_word == "DEF_OBJ") {
            scope = DEF_OBJECT;
            name = nth_word(line, 1);
            SceneAssets::Object object;
            object.name = name;
            objects.push_back(object);
        }
        else if (first_word == "ADD_SPHERE") {
            scope = DEF_SPHERE;
            name = get_unique_string();
            SceneAssets::Sphere sphere;
            sphere.name = name;
            spheres.push_back(sphere);   
        }
        else {
            //throw std::runtime_error("Unknown command: " + first_word);
            std::cout << "Unknown command: " << first_word << std::endl;
            exit(1);
        }

    }

    file.close();
}

void SceneParser::parseLineImageDef(const std::string& line, const std::string& name) {
    SceneAssets::Texture* texture = getTextureByName(name);
    std::string first_word = nth_word(line, 0);

    if(first_word == "PATH") {
        texture->path = nth_word(line, 1);
    } else {
        std::cout << "Unknown IMAGE definition command: " << first_word << std::endl;
        exit(1);
    }
}

void SceneParser::parseLineModelDef(const std::string& line, const std::string& name) {
    SceneAssets::Model* model = getModelByName(name);
    std::string first_word = nth_word(line, 0);

    if(first_word == "PATH") {
        model->path = nth_word(line, 1);
    }
    else {
        std::cout << "Unknown MODEL definition command: " << first_word << std::endl;
        exit(1);
    }
}

void SceneParser::parseLineMaterialDef(const std::string& line, const std::string& name) {
    SceneAssets::Material* material = getMaterialByName(name);
    std::string first_word = nth_word(line, 0);

    if (first_word == "TYPE") {
        material->type = static_cast<char>(std::stoi(nth_word(line, 1)));
    }
    else if(first_word == "COLOR") {
        material->color = Vec3(
            std::stof(nth_word(line, 1)),
            std::stof(nth_word(line, 2)),
            std::stof(nth_word(line, 3))
        );
    }
    else if(first_word == "P1") {
        material->p1 = std::stof(nth_word(line, 1));
    }
    else if(first_word == "P2") {
        material->p2 = std::stof(nth_word(line, 1));
    }
    else if(first_word == "P3") {
        material->p3 = std::stof(nth_word(line, 1));
    }
    else if (first_word == "TEXTURE") {
        if (material->image_names.size() < 3) {
            material->image_names.push_back(nth_word(line, 1));
        } else {
            std::cout << "Too many textures for material: " << name << std::endl;
            exit(1);
        }
    }
    else {
        std::cout << "Unknown MATERIAL definition command: " << first_word << std::endl;
        exit(1);
    }
}

void SceneParser::parseLineObjectDef(const std::string& line, const std::string& name) {
    SceneAssets::Object* object = getObjectByName(name);
    std::string first_word = nth_word(line, 0);

    if(first_word == "MODEL") {
        object->model_name = nth_word(line, 1);
    } 
    else if(first_word == "MATERIAL") {
        object->material_id = nth_word(line, 1);
    }
    else {
        std::cout << "Unknown OBJECT definition command: " << first_word << std::endl;
        exit(1);
    }

}

void SceneParser::parseLineObjectInstance(const std::string& line, const std::string& name) {
    SceneAssets::ObjectInstance* object_instance = getObjectInstanceByName(name);
    std::string first_word = nth_word(line, 0);
    if(first_word == "OBJECT") {
        object_instance->object_name = nth_word(line, 1);
    } 
    else if(first_word == "POSITION") {
        object_instance->position = Vec3(
            std::stof(nth_word(line, 1)),
            std::stof(nth_word(line, 2)),
            std::stof(nth_word(line, 3))
        );
    } 
    else if(first_word == "SCALE") {
        object_instance->scale = std::stof(nth_word(line, 1));
    }
    else if (first_word == "ROTATION") {
        object_instance->rotation = Vec3(
            std::stof(nth_word(line, 1)),
            std::stof(nth_word(line, 2)),
            std::stof(nth_word(line, 3))
        );
    }
    else {
        std::cout << "Unknown OBJECT INSTANCE definition command: " << first_word << std::endl;
        exit(1);
    }
}

void SceneParser::parseLineSphere(const std::string& line, const std::string& name) {
    if (!ENABLE_SPHERES) return;
    SceneAssets::Sphere* sphere = getSphereByName(name);
    std::string first_word = nth_word(line, 0);
    if(first_word == "POSITION") {
        sphere->position = Vec3(
            std::stof(nth_word(line, 1)),
            std::stof(nth_word(line, 2)),
            std::stof(nth_word(line, 3))
        );
    } 
    else if(first_word == "RADIUS") {
        sphere->radius = std::stof(nth_word(line, 1));
    } 
    else if(first_word == "MATERIAL") {
        sphere->material_id = nth_word(line, 1);
    }
    else {
        std::cout << "Unknown SPHERE definition command: " << first_word << std::endl;
        exit(1);
    }
}

const std::string SceneParser::nth_word(const std::string& line, int n) {
    size_t start = 0;
    size_t end = line.find(' ');
    int count = 0;

    while (end != std::string::npos) {
        if (count == n) {
            return line.substr(start, end - start);
        }
        start = end + 1;
        end = line.find(' ', start);
        count++;
    }

    if (count == n) {
        return line.substr(start);
    }

    return "";
}

SceneAssets::Texture* SceneParser::getTextureByName(std::string name) {
    for (auto& texture : textures) {
        if (texture.name == name) {
            return &texture;
        }
    }
    std::cout << "Texture not found: " << name << std::endl;
    exit(1);
}

SceneAssets::Model* SceneParser::getModelByName(std::string name) {
    for (auto& model : models) {
        if (model.name == name) {
            return &model;
        }
    }
    std::cout << "Model not found: " << name << std::endl;
    exit(1);
}

SceneAssets::Material* SceneParser::getMaterialByName(std::string name) {
    for (auto& material : materials) {
        if (material.name == name) {
            return &material;
        }
    }
    std::cout << "Material not found: " << name << std::endl;
    exit(1);
}

int SceneParser::getMaterialIndexByName(std::string name) {
    for (int i = 0; i < materials.size(); i++) {
        if (materials[i].name == name) {
            return i;
        }
    }
    std::cout << "Material not found: " << name << std::endl;
    exit(1);
}

SceneAssets::Object* SceneParser::getObjectByName(std::string name) {
    for (auto& object : objects) {
        if (object.name == name) {
            return &object;
        }
    }
    std::cout << "Object not found: " << name << std::endl;
    exit(1);
}

SceneAssets::ObjectInstance* SceneParser::getObjectInstanceByName(std::string name) {
    for (auto& object_instance : object_instances) {
        if (object_instance.name == name) {
            return &object_instance;
        }
    }
    std::cout << "Object Instance not found: " << name << std::endl;
    exit(1);
}

SceneAssets::Sphere* SceneParser::getSphereByName(std::string name) {
    for (auto& sphere : spheres) {
        if (sphere.name == name) {
            return &sphere;
        }
    }
    std::cout << "Sphere not found: " << name << std::endl;
    exit(1);
}

void SceneParser::getTriangleData(float** tris, size_t* arr_len, BVH::BVHNode** bvh_nodes, int* num_bvh_nodes) {  
    std::vector<Model> models;

    std::cout << "Number of object instances: " << object_instances.size() << std::endl;
    for (auto& object_instance : object_instances) {
        std::string object_name = object_instance.object_name;
        SceneAssets::Object* object = getObjectByName(object_name);
        int materialIndex = getMaterialIndexByName(object->material_id);
        std::string model_name = object->model_name;
        SceneAssets::Model* model_info = getModelByName(model_name);
        Model model;
        model.loadMesh(model_info->path.c_str(), object_instance.position, object_instance.scale, object_instance.rotation, materialIndex);
        models.push_back(std::move(model));
    }

    size_t total_length = 0;
    for (auto& model : models) {
        total_length += model.getDataLength();
    }

    int sphereDataSize = 0;
    if (ENABLE_SPHERES) {
        sphereDataSize = 1 + static_cast<int>(spheres.size()) * 5;
    }
    std::cout << "Total length of all triangle data: " << total_length << std::endl;
    std::cout << "Number of models: " << models.size() << std::endl;

    float *all_tris = new float[total_length + sphereDataSize];
    size_t offset = 0;

    for (auto& model : models) {
        float* model_tris = model.getFaces();
        size_t model_length = model.getDataLength();
        std::memcpy(all_tris + offset + sphereDataSize, model_tris, model_length * sizeof(float));
        offset += model_length;
    }

    BVH::createBVH(all_tris + sphereDataSize, total_length, bvh_nodes, num_bvh_nodes);
    int sphereOffset = 0;
    
    if (ENABLE_SPHERES) {
        all_tris[sphereOffset++] = static_cast<float>(spheres.size()) + 0.0001f;
        for (SceneAssets::Sphere& sphere : spheres) {
            int materialIndex = getMaterialIndexByName(sphere.material_id);
            all_tris[sphereOffset++] = sphere.position.x;
            all_tris[sphereOffset++] = sphere.position.y;
            all_tris[sphereOffset++] = sphere.position.z;
            all_tris[sphereOffset++] = sphere.radius;
            all_tris[sphereOffset++] = static_cast<float>(materialIndex) + 0.0001f;
        }
    }

    *tris = all_tris;
    *arr_len = total_length;
}

void SceneParser::getMaterialData(const Material** out_materials, size_t* num_materials, const float** out_texture, size_t* texture_length) {
    size_t material_count = materials.size();
    Material* material_array = new Material[material_count];
    std::vector<float> pixels = std::vector<float>();

    if (background_image_name != "") {
        SceneAssets::Texture* texture = getTextureByName(background_image_name);
        Texture tex(texture->path.c_str());
        texture->offset = pixels.size(); // 0
        const float* pixel_data = tex.getData();
        for (int i = 0; i < tex.dataLength(); i++) {
            pixels.push_back(pixel_data[i]);
        }
        texture->isLoaded = true;
        texture->width = tex.width;
        texture->height = tex.height;
        scene_configs.envTextureWidth = tex.width;
        scene_configs.envTextureHeight = tex.height;
    }

    for (int i = 0; i < material_count; i++) {
        material_array[i].type = materials[i].type;
        material_array[i].color = materials[i].color.colorToInt();
        material_array[i].p1 = materials[i].p1;
        material_array[i].p2 = materials[i].p2;
        material_array[i].p3 = materials[i].p3;

        for (int j = 0; j < materials[i].image_names.size(); j++) { // at most 3
            std::string image_name = materials[i].image_names[j];
            tryLoadTexture(material_array[i], j, image_name, pixels);
        }
    }

    float *texture_array = new float[pixels.size()];
    std::memcpy(texture_array, pixels.data(), pixels.size() * sizeof(float));
    *out_texture = texture_array;
    *texture_length = pixels.size();

    *out_materials = material_array;
    *num_materials = material_count;
}

void SceneParser::tryLoadTexture(Material& mat, int imageIndex, std::string imageName, std::vector<float>& pixels) {
    if(imageName == "NOISE") {
        int offset = pixels.size();
        mat.image1_offset = offset;
        mat.image_height = 512;
        mat.image_width = 512;

        NoiseTexture tex(mat.image_width, mat.image_height);
        tex.seed(37811); // big arbitrary prime
        tex.generate();

        for (int i = 0; i < mat.image_width * mat.image_height * 3; i++) {
            pixels.push_back(tex.data[i]);
        }

        return;
    }
    
    SceneAssets::Texture* texture = getTextureByName(imageName);
    if (!texture->isLoaded) {
        Texture tex(texture->path.c_str());
        texture->offset = pixels.size();
        const float* pixel_data = tex.getData();
        for (int i = 0; i < tex.dataLength(); i++) {
            pixels.push_back(pixel_data[i]);
        }
        texture->isLoaded = true;
        texture->width = tex.width;
        texture->height = tex.height;
    }

    if (imageIndex == 0) {
        mat.image1_offset = texture->offset;
    } else if (imageIndex == 1) {
        mat.image2_offset = texture->offset;
    } else if (imageIndex == 2) {
        mat.image3_offset = texture->offset;
    } else {
        std::cout << "Too many textures for material" << std::endl;
        exit(1);
    }

    if (mat.image_height != 0 && (mat.image_height != texture->height || mat.image_width != texture->width)) {
        std::cout << "Textures must have the same dimensions for a material." << std::endl;
        exit(1);
    } else {
        mat.image_height = texture->height;
        mat.image_width = texture->width;
    }
}

void SceneParser::getSceneConfigs(SceneConfigs* out_configs) {
    *out_configs = scene_configs;
}