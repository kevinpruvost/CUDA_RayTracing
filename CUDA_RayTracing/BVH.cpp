#include "BVH.h"

#include <algorithm>

BVH::BVH(Raytracer& raytracer)
    : root{ nullptr }
    , cuda_root{ nullptr } 
{
    Build(raytracer);
    LoadOnCUDA();
}

BVH::~BVH()
{
    Free();
    FreeOnCUDA();
}

bool isInside(Primitive* prim1, Primitive* prim2)
{
    return prim1->centroid.x >= prim2->min.x && prim1->centroid.x <= prim2->max.x &&
        prim1->centroid.y >= prim2->min.y && prim1->centroid.y <= prim2->max.y &&
        prim1->centroid.z >= prim2->min.z && prim1->centroid.z <= prim2->max.z;
}

struct MortonCode {
    uint32_t code;
    Primitive* primitive;
};

uint32_t computeMortonCode(const Vector3& point)
{
    uint32_t x = (uint32_t)(1023 * point.x);
    uint32_t y = (uint32_t)(1023 * point.y);
    uint32_t z = (uint32_t)(1023 * point.z);

    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    x = (x | (x << 2)) & 0x09249249;

    y = (y | (y << 16)) & 0x030000FF;
    y = (y | (y << 8)) & 0x0300F00F;
    y = (y | (y << 4)) & 0x030C30C3;
    y = (y | (y << 2)) & 0x09249249;

    z = (z | (z << 16)) & 0x030000FF;
    z = (z | (z << 8)) & 0x0300F00F;
    z = (z | (z << 4)) & 0x030C30C3;
    z = (z | (z << 2)) & 0x09249249;

    return x | (y << 1) | (z << 2);
}

std::vector<Primitive*> convertLinkedListToArray(Primitive* head) {
    std::vector<Primitive*> primitives;
    while (head) {
        primitives.push_back(head);
        head = head->next;
    }
    return primitives;
}

std::vector<MortonCode> computeAndSortMortonCodes(const std::vector<Primitive*>& primitives) {
    std::vector<MortonCode> mortonCodes;
    mortonCodes.reserve(primitives.size());

    for (auto* primitive : primitives) {
        Vector3 centroid = (primitive->min + primitive->max) * 0.5f;
        uint32_t code = computeMortonCode(centroid);
        mortonCodes.push_back({ code, primitive });
    }

    std::sort(mortonCodes.begin(), mortonCodes.end(), [](const MortonCode& a, const MortonCode& b) {
        return a.code < b.code;
        });

    return mortonCodes;
}

Vector3 min(const Vector3& a, const Vector3& b) {
    return Vector3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

Vector3 max(const Vector3& a, const Vector3& b) {
    return Vector3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

BVH_Node* buildBVHFromMortonCodes(const std::vector<MortonCode>& mortonCodes, int start, int end) {
    if (end - start == 1) {
        // Leaf node
        BVH_Node* leaf = new BVH_Node();
        leaf->min = mortonCodes[start].primitive->min;
        leaf->max = mortonCodes[start].primitive->max;
        leaf->primitive = mortonCodes[start].primitive;
        return leaf;
    }

    int split = (start + end) / 2;
    BVH_Node* node = new BVH_Node();
    node->left = buildBVHFromMortonCodes(mortonCodes, start, split);
    node->right = buildBVHFromMortonCodes(mortonCodes, split, end);

    node->min = min(node->left->min, node->right->min);
    node->max = max(node->left->max, node->right->max);
    return node;
}

void BVH::Build(Raytracer& raytracer)
{
    Primitive * primitive_head = raytracer.scene.primitive_head;

    auto primitives = convertLinkedListToArray(primitive_head);
    auto mortonCodes = computeAndSortMortonCodes(primitives);
    root = buildBVHFromMortonCodes(mortonCodes, 0, mortonCodes.size());
}

void BVH::Free()
{
}

void BVH::FreeOnCUDA()
{
}

void BVH::LoadOnCUDA()
{
}
