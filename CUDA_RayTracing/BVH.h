#pragma once

#include "primitive.h"
#include "Scene.cuh"
#include "raytracer.h"

struct BVH_Node
{
    BVH_Node() : left{ nullptr }, right{ nullptr }, primitive{ nullptr } {}

    BVH_Node * left;
    BVH_Node * right;
    Vector3 min;
    Vector3 max;
    Primitive * primitive;
};

class BVH
{
public:
    BVH(Raytracer & raytracer);
    ~BVH();

    void Build(Raytracer & raytracer);
    void Free();
    Cuda_BVH * LoadOnCUDA(BVH_Node * r);
    Cuda_BVH* LoadOnCUDA(Light* lights, Cuda_Light ** cudaLights, int lightCount);

    Cuda_Primitive * AllocateCudaPrimitive(Primitive* prim);

public:
    Light* __lights;
    Cuda_Light ** __cudaLights;
    int __lightCount;

    BVH_Node* _root;
    Cuda_BVH* cuda_root;
};

void FreeBVHOnCUDA(Cuda_BVH* node);
