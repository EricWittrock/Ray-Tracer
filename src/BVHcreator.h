#pragma once

#include "config.h"
#include "vec3.h"
#include <vector>
#include <memory>
#include <iostream>


namespace BVH 
{
    // temporary node used during construction only
    struct BVHTempNode {
        Vec3 bbox_min;
        Vec3 bbox_max;
        std::vector<const float*> triangles;
        std::unique_ptr<BVHTempNode> childA;
        std::unique_ptr<BVHTempNode> childB;
        int depth;
        int index;
    };

    struct BVHNode {
        float bbox_min_x;
        float bbox_min_y;
        float bbox_min_z;
        float bbox_max_x;
        float bbox_max_y;
        float bbox_max_z;
        int childAIndex; // -1 if leaf
        int childBIndex;
        int startTriDataOffs;
        int triDataLength;
    };


    BVHNode createBVHNodeFromTemp(BVHTempNode* tempNode);
    float getTriangleMaxPos(const float* tri, char axis);
    float getTriangleMinPos(const float* tri, char axis);
    void splitNode(BVHTempNode &node);
    BVHTempNode createInitialNode(const float* tris, size_t arr_len);
    void populateBVHNodes(BVHNode* bvhNodes, BVHTempNode* rootTempNode, float* newtris);
    void createBVH(float* tris, size_t arr_len, BVHNode** outNodes, int* outNumNodes);
}