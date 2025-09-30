#pragma once

#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include "vec3.h"
#include "triangle.h"

void loadModel(const std::string& filename, Triangle* triangles, int& numTriangles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error reading file: " << filename << std::endl;
        return;
    }
    std::string line;
    int triangleIndex = 0;

    std::vector<Vec3> verts;
    std::vector<Triangle> tris;

    while (std::getline(file, line)) {
        const char c0 = line[0];
        const char c1 = line[1];

        if (c0 == 'v' && c1 == ' ') {
            float x, y, z;
            sscanf(line.c_str(), "v %f %f %f", &x, &y, &z);
            verts.push_back(Vec3(x, y, z));
        }
        else if (c0 == 'f') {
            int v0, v1, v2;
            sscanf(line.c_str(), "f %d %d %d", &v0, &v1, &v2);
            tris.push_back(Triangle(verts[v0 - 1], verts[v1 - 1], verts[v2 - 1]));
        }
    }

    triangles = new Triangle[tris.size()];
    for (size_t i = 0; i < tris.size(); i++) {
        triangles[i] = tris[i];
    }
    numTriangles = tris.size();
}