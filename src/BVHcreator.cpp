#pragma once

#include "config.h"
#include "vec3.h"
#include <vector>
#include <memory>
#include <iostream>
#include "BVHcreator.h"


namespace BVH 
{
    BVHNode createBVHNodeFromTemp(BVHTempNode* tempNode) {
        BVHNode bvhNode;

        bvhNode.bbox_min_x = tempNode->bbox_min.x;
        bvhNode.bbox_min_y = tempNode->bbox_min.y;
        bvhNode.bbox_min_z = tempNode->bbox_min.z;
        bvhNode.bbox_max_x = tempNode->bbox_max.x;
        bvhNode.bbox_max_y = tempNode->bbox_max.y;
        bvhNode.bbox_max_z = tempNode->bbox_max.z;

        if (tempNode->childA) {
            bvhNode.childAIndex = tempNode->childA->index;
        } else {
            bvhNode.childAIndex = -1;
        }
        if (tempNode->childB) {
            bvhNode.childBIndex = tempNode->childB->index;
        } else {
            bvhNode.childBIndex = -1;
        }

        bvhNode.startTriDataOffs = 0; // to be filled in later
        bvhNode.triDataLength = static_cast<int>(tempNode->triangles.size()) * 25;

        return bvhNode;
    }

    // assign indices to nodes in BFS order. return number of nodes
    int BFSIndex(BVHTempNode* root) {
        std::vector<BVHTempNode*> queue;
        queue.push_back(root);
        int currentIndex = 0;

        while (!queue.empty()) {
            BVHTempNode* node = queue.front();
            queue.erase(queue.begin());
            node->index = currentIndex;
            currentIndex++;

            if (node->childA) {
                queue.push_back(node->childA.get());
            }
            if (node->childB) {
                queue.push_back(node->childB.get());
            }
        }

        return currentIndex;
    }


    float getTriangleMaxPos(const float* tri, char axis) {
        if (axis == 0) {
            return fmaxf(fmaxf(tri[0], tri[3]), tri[6]);
        } else if (axis == 1) {
            return fmaxf(fmaxf(tri[1], tri[4]), tri[7]);
        } else {
            return fmaxf(fmaxf(tri[2], tri[5]), tri[8]);
        }
    }


    float getTriangleMinPos(const float* tri, char axis) {
        if (axis == 0) {
            return fminf(fminf(tri[0], tri[3]), tri[6]);
        } else if (axis == 1) {
            return fminf(fminf(tri[1], tri[4]), tri[7]);
        } else {
            return fminf(fminf(tri[2], tri[5]), tri[8]);
        }
    }
    

    void splitNode(BVHTempNode &node) {        
        float maxPosA = -1e12f;
        float minPosB = 1e12f;
        char axis = 0; // 0=x, 1=y, 2=z
        float meanVertexPos = 0.0f;
        float xWidth = node.bbox_max.x - node.bbox_min.x;
        float yWidth = node.bbox_max.y - node.bbox_min.y;
        float zWidth = node.bbox_max.z - node.bbox_min.z;

        if(node.childA || node.childB) {
            std::cout << "Node already split" << std::endl;
            exit(1);
        }
        if (node.triangles.size() <= 2) {
            return;
        }
        node.childA = std::make_unique<BVHTempNode>();
        node.childB = std::make_unique<BVHTempNode>();
        node.childA->depth = node.depth + 1;
        node.childB->depth = node.depth + 1;

        if (xWidth >= yWidth && xWidth >= zWidth) {
            axis = 0;
        } else if (yWidth >= xWidth && yWidth >= zWidth) {
            axis = 1;
        } else {
            axis = 2;
        }

        for (size_t i = 0; i < node.triangles.size(); i++) {
            const float* tri = node.triangles[i];
            if (axis == 0) {
                meanVertexPos += (tri[0] + tri[3] + tri[6]);
            } else if (axis == 1) {
                meanVertexPos += (tri[1] + tri[4] + tri[7]);
            } else {
                meanVertexPos += (tri[2] + tri[5] + tri[8]);
            }
        }
        meanVertexPos /= static_cast<float>(node.triangles.size() * 3);

        for (size_t i = 0; i < node.triangles.size(); i++) {
            const float* tri = node.triangles[i];
            float centerPos = 0.0f;
            if (axis == 0) {
                centerPos = (tri[0] + tri[3] + tri[6]) / 3.0f;
            } else if (axis == 1) {
                centerPos = (tri[1] + tri[4] + tri[7]) / 3.0f;
            } else {
                centerPos = (tri[2] + tri[5] + tri[8]) / 3.0f;
            }

            if (centerPos < meanVertexPos) {
                node.childA->triangles.push_back(tri);
                maxPosA = fmaxf(maxPosA, getTriangleMaxPos(tri, axis));
            } else {
                node.childB->triangles.push_back(tri);
                minPosB = fminf(minPosB, getTriangleMinPos(tri, axis));
            }
        }

        node.childA->bbox_min = node.bbox_min.copy();
        node.childA->bbox_max = node.bbox_max.copy();
        node.childB->bbox_max = node.bbox_max.copy();
        node.childB->bbox_min = node.bbox_min.copy();
        if (axis == 0) {
            node.childA->bbox_max.x = maxPosA;
            node.childB->bbox_min.x = minPosB;
        } else if (axis == 1) {
            node.childA->bbox_max.y = maxPosA;
            node.childB->bbox_min.y = minPosB;
        } else {
            node.childA->bbox_max.z = maxPosA;
            node.childB->bbox_min.z = minPosB;
        }

        node.triangles.clear();
    }


    void splitRecursive(BVHTempNode *node) {
        if (node->depth >= BVH_DEPTH) {
            return;
        }
        splitNode(*node);
        if (node->childA) {
            splitRecursive(node->childA.get());
        }
        if (node->childB) {
            splitRecursive(node->childB.get());
        }
    }


    BVHTempNode createInitialNode(const float* tris, size_t arr_len) {
        BVHTempNode rootNode;
        rootNode.depth = 0;
        rootNode.bbox_min = Vec3(1e12f, 1e12f, 1e12f);
        rootNode.bbox_max = Vec3(-1e12f, -1e12f, -1e12f);
        for (size_t i = 0; i < arr_len; i += 25) {
            const float* tri = &tris[i];
            for (int v = 0; v < 3; v++) {
                rootNode.bbox_min.x = fminf(rootNode.bbox_min.x, tri[v * 3 + 0]);
                rootNode.bbox_min.y = fminf(rootNode.bbox_min.y, tri[v * 3 + 1]);
                rootNode.bbox_min.z = fminf(rootNode.bbox_min.z, tri[v * 3 + 2]);
                rootNode.bbox_max.x = fmaxf(rootNode.bbox_max.x, tri[v * 3 + 0]);
                rootNode.bbox_max.y = fmaxf(rootNode.bbox_max.y, tri[v * 3 + 1]);
                rootNode.bbox_max.z = fmaxf(rootNode.bbox_max.z, tri[v * 3 + 2]);
            }
            rootNode.triangles.push_back(tri);
        }
        return rootNode;
    }


        // call BFSIndex() first
    void populateBVHNodes(BVHNode* bvhNodes, BVHTempNode* rootTempNode, float* newtris) {
        std::vector<BVHTempNode*> queue;
        queue.push_back(rootTempNode);
        int triangleDataOffset = 0;

        while (!queue.empty()) {
            BVHTempNode* node = queue.front();
            queue.erase(queue.begin());

            BVHNode bvhNode = createBVHNodeFromTemp(node);
            bvhNode.startTriDataOffs = triangleDataOffset;
            int triCount = static_cast<int>(node->triangles.size());
            for (int i = 0; i < triCount; i++) {
                const float* tri = node->triangles[i];
                for (int v = 0; v < 25; v++) {
                    newtris[triangleDataOffset + i * 25 + v] = tri[v];
                }
            }
            triangleDataOffset += triCount * 25;
            bvhNodes[node->index] = bvhNode;

            if (node->childA) {
                queue.push_back(node->childA.get());
            }
            if (node->childB) {
                queue.push_back(node->childB.get());
            }
        }
    }


    void createBVH(float* tris, size_t arr_len, BVHNode** outNodes, int* outNumNodes) {
        BVHTempNode rootNode = createInitialNode(tris, arr_len);
        
        splitRecursive(&rootNode);
        int numNodes = BFSIndex(&rootNode);
        BVHNode *bvhNodes = new BVHNode[numNodes];
        float* newtris = new float[arr_len];
        populateBVHNodes(bvhNodes, &rootNode, newtris);

        // copy newtris back to tris
        for (size_t i = 0; i < arr_len; i++) {
            tris[i] = newtris[i];
        }

        delete[] newtris;
        *outNodes = bvhNodes;
        *outNumNodes = numNodes;


        // BVHTempNode rootNode = createInitialNode(tris, arr_len);

        // BVHNode *bvhNodes = new BVHNode[1];
        // bvhNodes[0] = createBVHNodeFromTemp(&rootNode);
        // bvhNodes[0].startTriDataOffs = 0;
        // bvhNodes[0].triDataLength = arr_len;

        // *outNodes = bvhNodes;
        // *outNumNodes = 1;
    }
}