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
    float3 color;
    float3 O;
    float3 Dx;
    float3 Dy;
    double R;
    Cuda_Light_Type type;
    Cuda_Primitive* lightPrimitive;
};

#endif