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
    size_t facesLength;
public:
    __host__ Model() : faces(nullptr), facesLength(0) {}

    __host__ ~Model() {
        if (faces) {
            delete[] faces;
        }
    }

    // use with std::move
    Model(Model&& other) : faces(other.faces), facesLength(other.facesLength) {
        other.faces = nullptr;
        other.facesLength = 0;
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

    __host__ void loadMesh(const char* filename, Vec3& translation) 
    {
        std::cout << "before all" << std::endl;

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cout << "Could not open model file: " << filename << std::endl;
            exit(1);
        }
        std::string line;

        std::vector<float> verts;
        std::vector<float> norms;
        std::vector<float> uvs;
        std::vector<int> tri_indices;
        std::vector<float> tris;

        std::cout << "before while" << std::endl;

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

        std::cout << "after while" << std::endl;

        file.close();

        std::cout << "after close" << std::endl;

        for (size_t i = 0; i < tri_indices.size(); i += 9) {
            int v0_idx = (tri_indices[i] - 1) * 3;
            int v1_idx = (tri_indices[i + 1] - 1) * 3;
            int v2_idx = (tri_indices[i + 2] - 1) * 3;
            Vec3 v0(verts[v0_idx], verts[v0_idx + 1], verts[v0_idx + 2]);
            Vec3 v1(verts[v1_idx], verts[v1_idx + 1], verts[v1_idx + 2]);
            Vec3 v2(verts[v2_idx], verts[v2_idx + 1], verts[v2_idx + 2]);

            int n0_idx = (tri_indices[i + 3] - 1) * 3;
            int n1_idx = (tri_indices[i + 4] - 1) * 3;
            int n2_idx = (tri_indices[i + 5] - 1) * 3;
            Vec3 n0(norms[n0_idx], norms[n0_idx + 1], norms[n0_idx + 2]);
            Vec3 n1(norms[n1_idx], norms[n1_idx + 1], norms[n1_idx + 2]);
            Vec3 n2(norms[n2_idx], norms[n2_idx + 1], norms[n2_idx + 2]);

            int t0_idx = (tri_indices[i + 6] - 1) * 2;
            int t1_idx = (tri_indices[i + 7] - 1) * 2;
            int t2_idx = (tri_indices[i + 8] - 1) * 2;
            Vec3 uv0(uvs[t0_idx], uvs[t0_idx + 1], 0);
            Vec3 uv1(uvs[t1_idx], uvs[t1_idx + 1], 0);
            Vec3 uv2(uvs[t2_idx], uvs[t2_idx + 1], 0);

            tris.push_back(v0.x);
            tris.push_back(v0.y);
            tris.push_back(v0.z);
            tris.push_back(v1.x);
            tris.push_back(v1.y);
            tris.push_back(v1.z);
            tris.push_back(v2.x);
            tris.push_back(v2.y);
            tris.push_back(v2.z);

            tris.push_back(n0.x);
            tris.push_back(n0.y);
            tris.push_back(n0.z);
            tris.push_back(n1.x);
            tris.push_back(n1.y);
            tris.push_back(n1.z);
            tris.push_back(n2.x);
            tris.push_back(n2.y);
            tris.push_back(n2.z);

            tris.push_back(uv0.x);
            tris.push_back(uv0.y);
            tris.push_back(uv1.x);
            tris.push_back(uv1.y);
            tris.push_back(uv2.x);
            tris.push_back(uv2.y);
        }

        facesLength = tris.size();
        std::cout << "facesLength: " << facesLength << std::endl;
        if (faces) {
            delete[] faces;
            faces = nullptr;
        }
        if (facesLength > 0) {
            faces = new float[facesLength];
            for (size_t i = 0; i < facesLength; i++) {
                faces[i] = tris[i];
            }
        } else {
            faces = nullptr;
        }
        faces[facesLength] = 123.0f;
        std::cout << "done: " << facesLength << std::endl;
    }
};