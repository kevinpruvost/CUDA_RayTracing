#pragma once

#include "primitive.h"
#include "Scene.cuh"
#include "raytracer.h"

struct BVH_Node
{
    BVH_Node() : left{ nullptr }, right{ nullptr } {}

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
    void FreeOnCUDA();
    void LoadOnCUDA();

public:
    BVH_Node* root;
    Cuda_BVH* cuda_root;
};

