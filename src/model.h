#pragma once

#include "triangle.h"
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include "vec3.h"
#include "object.h"

class Model {
private:
    float* faces;
    unsigned int facesLength;
public:
    __host__ Model() : faces(nullptr), facesLength(0) {}

    __host__ ~Model() {
        if (faces) {
            delete[] faces;
        }
    }

    __host__ float* getFaces() const {
        return faces;
    }

    __host__ unsigned int getNumTris() const {
        return facesLength / 9;
    }

    __host__ unsigned int getDataLength() const {
        return facesLength;
    }

    __host__ size_t getSizeBytes() const {
        return getDataLength() * sizeof(float);
    }

    __host__ void loadMesh(const char* filename, Vec3 translation) 
    {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error reading file: " << filename << std::endl;
            return;
        }
        std::string line;

        std::vector<float> verts;
        std::vector<float> norms;
        std::vector<float> uvs;
        std::vector<int> tri_indices;
        std::vector<float> tris;

        while (std::getline(file, line)) {
            const char c0 = line[0];
            const char c1 = line[1];

            if (c0 == 'v' && c1 == ' ') {
                float x, y, z;
                sscanf(line.c_str(), "v %f %f %f", &x, &y, &z);
                verts.push_back(x + translation.x);
                verts.push_back(y + translation.y);
                verts.push_back(z + translation.z);
            }
            else if (c0 == 'v' && c1 == 'n') {
                float x, y, z;
                sscanf(line.c_str(), "vn %f %f %f", &x, &y, &z);
                norms.push_back(x);
                norms.push_back(y);
                norms.push_back(z);
            }
            else if (c0 == 'v' && c1 == 't') {
                float u, v;
                sscanf(line.c_str(), "vt %f %f", &u, &v);
                uvs.push_back(u);
                uvs.push_back(v);
            }
            else if (c0 == 'f' && c1 == ' ') {
                int v0, v1, v2; // vertex indices
                int t0, t1, t2; // texture coord indices
                int n0, n1, n2; // normal indices
                sscanf(line.c_str(), "f %d/%d/%d %d/%d/%d %d/%d/%d", &v0, &t0, &n0, &v1, &t1, &n1, &v2, &t2, &n2);
                tri_indices.push_back(v0);
                tri_indices.push_back(v1);
                tri_indices.push_back(v2);
                tri_indices.push_back(n0);
                tri_indices.push_back(n1);
                tri_indices.push_back(n2);
                tri_indices.push_back(t0);
                tri_indices.push_back(t1);
                tri_indices.push_back(t2);
            }
        }

        file.close();

        for (int i = 0; i < tri_indices.size(); i += 9) {
            int v0_idx = (tri_indices[i] - 1) * 3;
            int v1_idx = (tri_indices[i + 1] - 1) * 3;
            int v2_idx = (tri_indices[i + 2] - 1) * 3;

            Vec3 v0(verts[v0_idx], verts[v0_idx + 1], verts[v0_idx + 2]);
            Vec3 v1(verts[v1_idx], verts[v1_idx + 1], verts[v1_idx + 2]);
            Vec3 v2(verts[v2_idx], verts[v2_idx + 1], verts[v2_idx + 2]);

            tris.push_back(v0.x);
            tris.push_back(v0.y);
            tris.push_back(v0.z);
            tris.push_back(v1.x);
            tris.push_back(v1.y);
            tris.push_back(v1.z);
            tris.push_back(v2.x);
            tris.push_back(v2.y);
            tris.push_back(v2.z);
        }

        facesLength = tris.size();
        faces = new float[facesLength];
        for (size_t i = 0; i < facesLength; i++) {
            faces[i] = tris[i];
        }
    }
};