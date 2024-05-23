#ifndef LIGHT_CUH
#define LIGHT_CUH

#include <cuda_runtime.h>

#include "Primitive.cuh"

enum Cuda_Light_Type
{
    Cuda_Light_Type_Point,
    Cuda_Light_Type_Square,
    Cuda_Light_Type_Sphere
};

struct Cuda_Light
{
    int sample;
    double3 color;
    double3 O;
    double3 Dx;
    double3 Dy;
    double R;
    Cuda_Light_Type type;
    Cuda_Primitive* lightPrimitive;
};

__device__ double3 GetRandPointLight(double3 C, Cuda_Light* light);
__device__ double CalnShade(double3 C, Cuda_Primitive* crashed_Primitive, Cuda_Light* light, Cuda_BVH * bvh, int shade_quality);
__device__ Cuda_Collision intersect(Cuda_BVH* bvh, const double3* origin, const double3* direction, Cuda_Primitive ** ignorePrimitive = nullptr);

#endif