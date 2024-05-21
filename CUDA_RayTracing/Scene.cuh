#ifndef SCENE_CUH
#define SCENE_CUH

#include <cuda_runtime.h>

#include "Light.cuh"
#include "Camera.cuh"

struct Cuda_Scene
{
    double3 backgroundColor_top;
    double3 backgroundColor_bottom;
    Cuda_Camera camera;
    Cuda_Light* lights;
    int lightCount;
    Cuda_Primitive * primitives;
    int primitiveCount;
    double3 * resampling_surface;
};

__device__ double3 traceRay(Cuda_Scene * scene, double x, double y, int depth);

#endif // !SCENE_CUH