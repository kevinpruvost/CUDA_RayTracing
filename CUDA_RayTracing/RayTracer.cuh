#pragma once

#include <cuda_runtime.h>

// Declare the CUDA kernel
__global__ void rayTraceKernel(uchar4* output, int width, int height);
// Declare the wrapper function
void launchRayTraceKernel(uchar4* d_output, int width, int height);

// Utility functions (if needed, you can also declare them here)
__device__ float3 make_float3(float x, float y, float z);
__device__ float3 normalize(const float3& v);
__device__ float3 lerp(const float3& a, const float3& b, float t);
__device__ float3 traceRay(const float3& origin, const float3& direction);