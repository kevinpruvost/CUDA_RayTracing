#ifndef RAYTRACER_CUH
#define RAYTRACER_CUH

#include <cuda_runtime.h>
#include "Scene.cuh"

#define RESAMPLING_SIZE 8 // N for NxN super sampling

// Declare the CUDA kernel
__global__ void rayTraceKernel(uchar4* output, int width, int height);
// Declare the wrapper function
void launchRayTraceKernel(cudaSurfaceObject_t surface, int texture_width, int texture_height, int viewport_width, int viewport_height, Cuda_Scene* scene, int x_progress, int y_progress, int width_to_do, int height_to_do, Settings* settings);
// Utility functions (if needed, you can also declare them here)
//__device__ float3 make_float3(float x, float y, float z);
//__device__ float3 normalize(const float3& v);
//__device__ float3 lerp(const float3& a, const float3& b, float t);
//__device__ float3 traceRay(const float3& origin, const float3& direction);

#endif // !RAYTRACER_CUH