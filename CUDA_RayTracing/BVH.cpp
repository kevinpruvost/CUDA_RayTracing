#include "BVH.h"

#include <algorithm>
#include <assert.h>

#ifndef _DEBUG
#undef assert
#define assert(x) if (!(x)) { printf("Assertion failed: %s\n", #x); exit(1); }
#endif

BVH::BVH(Raytracer& raytracer)
    : _root{ nullptr }
    , cuda_root{ nullptr } 
{
    Build(raytracer);
}

BVH::~BVH()
{
    Free();
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

    if (primitive_head == nullptr) return;
    auto primitives = convertLinkedListToArray(primitive_head);
    auto mortonCodes = computeAndSortMortonCodes(primitives);
    _root = buildBVHFromMortonCodes(mortonCodes, 0, mortonCodes.size());
}

void FreeNode(BVH_Node* node)
{
    if (!node) return;

    FreeNode(node->left);
    FreeNode(node->right);
    delete node;
}

void BVH::Free()
{
    if (!_root) return;

    FreeNode(_root);
}

#define MALLOC(ptr, size) assert(cudaMalloc(ptr, size) == cudaSuccess)
#define FREE(ptr) assert(cudaFree(ptr) == cudaSuccess)
#define MEMCPY(dst, src, size, kind) assert(cudaMemcpy(dst, src, size, kind) == cudaSuccess)

void FreeBVHOnCUDA(Cuda_BVH* node)
{
    if (!node) return;

    Cuda_BVH* otherNode;
    MEMCPY(&otherNode, &node->left, sizeof(Cuda_BVH*), cudaMemcpyDeviceToHost);
    if (otherNode)
    {
        FreeBVHOnCUDA(otherNode);
    }
    Cuda_BVH* otherNode2;
    MEMCPY(&otherNode2, &node->right, sizeof(Cuda_BVH*), cudaMemcpyDeviceToHost);
    if (otherNode2)
    {
        FreeBVHOnCUDA(otherNode2);
    }
    double3 min, max;
    MEMCPY(&min, &node->min, sizeof(double3), cudaMemcpyDeviceToHost);
    MEMCPY(&max, &node->max, sizeof(double3), cudaMemcpyDeviceToHost);
    Cuda_Primitive* prim;
    MEMCPY(&prim, &node->primitive, sizeof(Cuda_Primitive*), cudaMemcpyDeviceToHost);
    if (prim)
    {
        Cuda_Primitive data;
        MEMCPY(&data, prim, sizeof(Cuda_Primitive), cudaMemcpyDeviceToHost);
        if (data.material.texture) FREE(data.material.texture);
        FREE(prim);
    }
    FREE(node);
}

Cuda_BVH * BVH::LoadOnCUDA(BVH_Node * root)
{
    Cuda_BVH* cuda_root;
    MALLOC(&cuda_root, sizeof(Cuda_BVH));
    cudaMemset(cuda_root, 0, sizeof(Cuda_BVH));

    Cuda_BVH* new_node;
    if (root->left) {
        new_node = LoadOnCUDA(root->left);
        MEMCPY(&cuda_root->left, &new_node, sizeof(Cuda_BVH*), cudaMemcpyHostToDevice);
    }

    if (root->right) {
        new_node = LoadOnCUDA(root->right);
        MEMCPY(&cuda_root->right, &new_node, sizeof(Cuda_BVH*), cudaMemcpyHostToDevice);
    }

    MEMCPY(&cuda_root->min, &root->min, sizeof(double3), cudaMemcpyHostToDevice);
    MEMCPY(&cuda_root->max, &root->max, sizeof(double3), cudaMemcpyHostToDevice);
    
    Cuda_Primitive * primitive = 0;
    if (root->primitive) {
        primitive = AllocateCudaPrimitive(root->primitive);
    }
    MEMCPY(&cuda_root->primitive, &primitive, sizeof(Cuda_Primitive*), cudaMemcpyHostToDevice);
    return cuda_root;
}

Cuda_BVH* BVH::LoadOnCUDA(Light* lights, Cuda_Light ** cudaLights, int lightCount)
{
    __lights = lights;
    __cudaLights = cudaLights;
    __lightCount = lightCount;
    return LoadOnCUDA(_root);
}

int Count(BVH_Node* node)
{
    if (!node) return 0;
    return ((node->primitive) ? 1 : 0)+ Count(node->left) + Count(node->right);
}

int BVH::Size()
{
    return Count(_root);
}

Cuda_Primitive* BVH::AllocateCudaPrimitive(Primitive* currentPrimitive)
{
    Cuda_Primitive* cudaPrimitive;
    MALLOC(&cudaPrimitive, sizeof(Cuda_Primitive));
    // Primitive properties
    bool isLightPrimitive = currentPrimitive->IsLightPrimitive();
    MEMCPY(&cudaPrimitive->isLightPrimitive, &isLightPrimitive, sizeof(bool), cudaMemcpyHostToDevice);
    // Material
    {
        //struct Cuda_Material
        //{
        //    float3 color, absor;
        //    double refl, refr;
        //    double diff, spec;
        //    double rindex;
        //    double drefl;
        //    uchar3 * texture;
        //    int texture_width, texture_height;
        //};
        Material* material = currentPrimitive->GetMaterial();
        MEMCPY(&cudaPrimitive->material.color, &material->color, sizeof(Color), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->material.absor, &material->absor, sizeof(Color), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->material.refl, &material->refl, sizeof(double), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->material.refr, &material->refr, sizeof(double), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->material.diff, &material->diff, sizeof(double), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->material.spec, &material->spec, sizeof(double), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->material.rindex, &material->rindex, sizeof(double), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->material.drefl, &material->drefl, sizeof(double), cudaMemcpyHostToDevice);
        if (material->texture != nullptr) {
            int texture_width = material->texture->GetW();
            int texture_height = material->texture->GetH();
            MEMCPY(&cudaPrimitive->material.texture_height, &texture_height, sizeof(int), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitive->material.texture_width, &texture_width, sizeof(int), cudaMemcpyHostToDevice);

            uchar3* texture = new uchar3[texture_width * texture_height];
            for (int i = 0; i < texture_width; ++i)
                for (int j = 0; j < texture_height; ++j)
                    memcpy(&texture[i * (int)texture_height + j], &material->texture->ima[j][i], sizeof(uchar3));
            uchar3* texturePtr;
            MALLOC(&texturePtr, texture_width * texture_height * sizeof(uchar3));
            MEMCPY(texturePtr, texture, texture_width * texture_height * sizeof(uchar3), cudaMemcpyHostToDevice);
            MEMCPY(&(cudaPrimitive->material.texture), &texturePtr, sizeof(uchar3*), cudaMemcpyHostToDevice);
            delete texture;
        }
        else {
            cudaMemset(&cudaPrimitive->material.texture, 0, sizeof(uchar3**));
            cudaMemset(&cudaPrimitive->material.texture_width, 0, sizeof(double));
            cudaMemset(&cudaPrimitive->material.texture_height, 0, sizeof(double));
        }
    }

    // Primitive type
    Cuda_Primitive_Type type;
    if (dynamic_cast<Sphere*>(currentPrimitive) != nullptr)
    {
        Sphere* sphere = dynamic_cast<Sphere*>(currentPrimitive);
        MEMCPY(&cudaPrimitive->data.sphere.O, &sphere->O, sizeof(Vector3), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.sphere.R, &sphere->R, sizeof(double), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.sphere.De, &sphere->De, sizeof(Vector3), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.sphere.Dc, &sphere->Dc, sizeof(Vector3), cudaMemcpyHostToDevice);
        type = Cuda_Primitive_Type_Sphere;
    }
    else if (dynamic_cast<Plane*>(currentPrimitive) != nullptr)
    {
        Plane* plane = dynamic_cast<Plane*>(currentPrimitive);
        MEMCPY(&cudaPrimitive->data.plane.N, &plane->N, sizeof(Vector3), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.plane.R, &plane->R, sizeof(double), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.plane.Dx, &plane->Dx, sizeof(Vector3), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.plane.Dy, &plane->Dy, sizeof(Vector3), cudaMemcpyHostToDevice);
        type = Cuda_Primitive_Type_Plane;
    }
    else if (dynamic_cast<Square*>(currentPrimitive) != nullptr)
    {
        Square* square = dynamic_cast<Square*>(currentPrimitive);
        MEMCPY(&cudaPrimitive->data.square.O, &square->O, sizeof(Vector3), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.square.Dx, &square->Dx, sizeof(Vector3), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.square.Dy, &square->Dy, sizeof(Vector3), cudaMemcpyHostToDevice);
        type = Cuda_Primitive_Type_Square;
    }
    else if (dynamic_cast<Cylinder*>(currentPrimitive) != nullptr)
    {
        Cylinder* cylinder = dynamic_cast<Cylinder*>(currentPrimitive);
        MEMCPY(&cudaPrimitive->data.cylinder.O1, &cylinder->O1, sizeof(Vector3), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.cylinder.O2, &cylinder->O2, sizeof(Vector3), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.cylinder.R, &cylinder->R, sizeof(double), cudaMemcpyHostToDevice);
        type = Cuda_Primitive_Type_Cylinder;
    }
    else if (dynamic_cast<Bezier*>(currentPrimitive) != nullptr)
    {
        Bezier* bezier = dynamic_cast<Bezier*>(currentPrimitive);
        MEMCPY(&cudaPrimitive->data.bezier.O1, &bezier->O1, sizeof(Vector3), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.bezier.O2, &bezier->O2, sizeof(Vector3), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.bezier.degree, &bezier->degree, sizeof(int), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.bezier.N, &bezier->N, sizeof(Vector3), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.bezier.Nx, &bezier->Nx, sizeof(Vector3), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.bezier.Ny, &bezier->Ny, sizeof(Vector3), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.bezier.R_c, &bezier->R_c, sizeof(double), cudaMemcpyHostToDevice);
        double* rData = bezier->R.data();
        double* zData = bezier->Z.data();
        MEMCPY(&cudaPrimitive->data.bezier.R, rData, (bezier->degree + 1) * sizeof(double), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.bezier.Z, zData, (bezier->degree + 1) * sizeof(double), cudaMemcpyHostToDevice);

        type = Cuda_Primitive_Type_Bezier;
    }
    else if (dynamic_cast<Triangle*>(currentPrimitive) != nullptr)
    {
        type = Cuda_Primitive_Type_Triangle;

        Triangle* triangle = dynamic_cast<Triangle*>(currentPrimitive);
        MEMCPY(&cudaPrimitive->data.triangle.O1, &triangle->O1, sizeof(Vector3), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.triangle.O2, &triangle->O2, sizeof(Vector3), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.triangle.O3, &triangle->O3, sizeof(Vector3), cudaMemcpyHostToDevice);
        MEMCPY(&cudaPrimitive->data.triangle.N, &triangle->N, sizeof(Vector3), cudaMemcpyHostToDevice);
    }
    else
    {
        assert(false);
    }
    MEMCPY(&cudaPrimitive->type, &type, sizeof(Cuda_Primitive_Type), cudaMemcpyHostToDevice);

    // Link light primitives to their lights
    Cuda_Light * cudaLights;
    MEMCPY(&cudaLights, __cudaLights, sizeof(Cuda_Light *), cudaMemcpyDeviceToHost);
    if (isLightPrimitive) {
        Light* light = __lights;
        for (int i = 0; i < __lightCount; ++i) {
            if (__lights->lightPrimitive == currentPrimitive) {
                MEMCPY(&cudaLights[i].lightPrimitive, &cudaLights[i], sizeof(Cuda_Light), cudaMemcpyHostToDevice);
                break;
            }
            light = light->next;
        }
    }

    return cudaPrimitive;
}
