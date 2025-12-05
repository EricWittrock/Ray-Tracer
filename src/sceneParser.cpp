#include "sceneParser.h"
#include "model.h"

namespace {
    enum Scope {
        GLOBAL,
        DEF_IMAGE,
        DEF_MODEL,
        DEF_MATERIAL,
        DEF_OBJECT,
        DEF_OBJECT_INSTANCE
    };
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
        std::cout << "line: " << line << std::endl;
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
        }


        std::string first_word = nth_word(line, 0);

        if (first_word == "SET_IMAGE_WIDTH") {
            image_width = std::stoi(nth_word(line, 1));
        } 
        else if (first_word == "SET_NUM_SAMPLES") {
            num_samples = std::stoi(nth_word(line, 1));
        } 
        else if (first_word == "SET_MAX_BOUNCES") {
            max_bounces = std::stoi(nth_word(line, 1));
        } 
        else if (first_word == "SET_CAM_POS") {
            camera_pos = Vec3(
                std::stof(nth_word(line, 1)),
                std::stof(nth_word(line, 2)),
                std::stof(nth_word(line, 3))
            );
        } 
        else if (first_word == "SET_CAM_LOOK_AT") {
            camera_look_at = Vec3(
                std::stof(nth_word(line, 1)),
                std::stof(nth_word(line, 2)),
                std::stof(nth_word(line, 3))
            );
        } 
        else if (first_word == "SET_FOCAL_LENGTH") {
            focal_length = std::stof(nth_word(line, 1));
        } 
        else if (first_word == "SET_BACKGROUND") {
            background_texture_name = nth_word(line, 1);
        } 
        else if (first_word == "ADD_OBJECT") {
            scope = DEF_OBJECT_INSTANCE;
            name = nth_word(line, 1);
            std::cout << "test ADD_OBJECT, " << name << std::endl;
            SceneAssets::ObjectInstance object_instance;
            object_instance.name = name;
            object_instances.push_back(object_instance);

            for (auto& oi : object_instances) {
                std::cout << "Existing object instance: " << oi.name << std::endl;
            }
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
            std::cout << "test" << std::endl;
            scope = DEF_MATERIAL;
            name = nth_word(line, 1);
            SceneAssets::Material material;
            material.name = name;
            materials.push_back(material);
        } 
        else if (first_word == "DEF_OBJ") {
            std::cout << "test" << std::endl;
            scope = DEF_OBJECT;
            name = nth_word(line, 1);
            SceneAssets::Object object;
            object.name = name;
            objects.push_back(object);
        } else {
            //throw std::runtime_error("Unknown command: " + first_word);
            std::cout << "Unknown command: " << first_word << std::endl;
            exit(1);
        }

    }

    file.close();
}

void SceneParser::parseLineImageDef(const std::string& line, const std::string& name) {
    std::cout << "test" << std::endl;
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
    std::cout << "test" << std::endl;
    SceneAssets::Model* model = getModelByName(name);
    std::string first_word = nth_word(line, 0);

    if(first_word == "PATH") {
        model->path = nth_word(line, 1);
    } 
    else if(first_word == "MTL_PATH") {
        model->mtl_path = nth_word(line, 1);
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
    else {
        std::cout << "Unknown OBJECT INSTANCE definition command: " << first_word << std::endl;
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
        model.loadMesh(model_info->path.c_str(), object_instance.position, materialIndex);
        models.push_back(std::move(model));
    }

    size_t total_length = 0;
    for (auto& model : models) {
        total_length += model.getDataLength();
    }
    std::cout << "Total length of all triangle data: " << total_length << std::endl;
    std::cout << "Number of models: " << models.size() << std::endl;

    float *all_tris = new float[total_length];
    size_t offset = 0;
    for (auto& model : models) {
        float* model_tris = model.getFaces();
        size_t model_length = model.getDataLength();
        std::memcpy(all_tris + offset, model_tris, model_length * sizeof(float));
        offset += model_length;
    }

    BVH::createBVH(all_tris, total_length, bvh_nodes, num_bvh_nodes);
    
    *tris = all_tris;
    *arr_len = total_length;
}

void SceneParser::getMaterialData(const Material** out_materials, size_t* num_materials) {
    size_t count = materials.size();
    Material* material_array = new Material[count];

    for (int i = 0; i < count; i++) {
        material_array[i].type = materials[i].type;
        material_array[i].color = materials[i].color.colorToInt();
        material_array[i].p1 = materials[i].p1;
        material_array[i].p2 = materials[i].p2;
        material_array[i].p3 = materials[i].p3;
        // material_array[i].albedo = this->materials[i].albedo;
    }

    *out_materials = material_array;
    *num_materials = count;
}