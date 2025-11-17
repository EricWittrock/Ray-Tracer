#pragma once

#include "vec3.h"
#include "vector"
#include <fstream>
#include <string>

class SceneConfigs {
public:
    Vec3 camera_pos;
    Vec3 camera_forward;
    float focal_length;
    int num_samples;
    int max_bounces;
    int image_width;
    int num_objects;
    int num_assets;
    std::vector<const char*> texture_paths;


    SceneConfigs()
        : camera_pos(0.0f, 0.0f, 0.0f),
          camera_forward(0.0f, 0.0f, -1.0f),
          focal_length(0.3f),
          num_samples(1),
          max_bounces(1),
          image_width(800),
          num_objects(0),
          num_assets(0) {}


    void parseFromFile(const char* path) {
        std::ifstream file(path);
        std::string line;

        if (!file.is_open()) {
            throw std::runtime_error("Could not open config file.");
        }

        // 0 = global, 1 = def_material, 2 = def_object
        int scope = 0;

        while (std::getline(file, line)) {
            if (line.at(0) == '#') continue; // Ignore comments
            std::string first_word = nth_word(line, 0);
            if (first_word == "DEF_MATERIAL") {
                scope = 1;
            } else if (first_word == "DEF_OBJ") {
                scope = 2;
            } else if (first_word == "END") {
                scope = 0;
            } else {
                parseLine(line);
            }
        }

        file.close();
    }

    void parseLine(const std::string& line) {
        
        std::string first_word = nth_word(line, 0);
    }

    const std::string nth_word(const std::string& line, int n) {
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
};
