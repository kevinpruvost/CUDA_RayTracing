#ifndef SCENE_CUH
#define SCENE_CUH

#include <cuda_runtime.h>

#include "Light.cuh"
#include "Camera.cuh"

struct Cuda_Scene
{
    float3 backgroundColor_top;
    float3 backgroundColor_bottom;
    Cuda_Camera camera;
    Cuda_Light* lights;
    int lightCount;
    Cuda_Primitive * primitives;
    int primitiveCount;
};

__device__ float3 traceRay(Cuda_Scene * scene, double u, double v);

#endif // !SCENE_CUH