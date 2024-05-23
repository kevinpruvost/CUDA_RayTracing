#ifndef SCENE_CUH
#define SCENE_CUH

#include <cuda_runtime.h>

#include "Light.cuh"
#include "Camera.cuh"

struct DepthOfField
{
    bool enabled;
    double focalDistance;
    double aperture;
};

struct Settings
{
    DepthOfField depthOfField;
    int resampling_size;
};

struct Cuda_Scene
{
    double3 backgroundColor_top;
    double3 backgroundColor_bottom;
    Cuda_Camera camera;
    Cuda_Light* lights;
    int lightCount;
    Cuda_BVH* bvh;
//    Cuda_Primitive * primitives;
    int primitiveCount;
    unsigned long * seeds;
};

__device__ bool traversBVH(Cuda_BVH * node, const double3 * origin, const double3 * direction, Cuda_Collision * collision);
__device__ double3 traceRay(Cuda_Scene * scene, double3 origin, double3 direction, int depth);
__device__ double2 GetBlur();
__device__ double3 CalnDiffusion(Cuda_Scene* scene, Cuda_Collision* collide_primitive);
__device__ double3 CalnReflection(Cuda_Scene * scene, Cuda_Collision * collide_primitive, double3 ray_V, int dep);
__device__ double3 CalnRefraction(Cuda_Scene* scene, Cuda_Collision* collide_primitive, double3 ray_V, int dep);

#endif // !SCENE_CUH