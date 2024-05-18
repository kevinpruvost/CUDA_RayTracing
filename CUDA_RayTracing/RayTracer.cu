#include <cuda_runtime.h>
#include <math_functions.h>
#include <device_launch_parameters.h>
#include <iostream>

// Utility function to normalize a float3
__device__ float3 normalize(const float3& v) {
    float len = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return {
        v.x / len,
        v.y / len,
        v.z / len
    };
}

// Utility function to perform linear interpolation
__device__ float3 lerp(const float3& a, const float3& b, float t) {
    return {
        a.x + t * (b.x - a.x),
        a.y + t * (b.y - a.y),
        a.z + t * (b.z - a.z)
    };
}

// Simple background color function
__device__ float3 traceRay(const float3& origin, const float3& direction) {
    // Linearly interpolate between white and blue based on the y coordinate
    float t = 0.5f * (direction.y + 1.0f);
    return lerp({ 1.0f, 1.0f, 1.0f }, { 0.5f, 0.7f, 1.0f }, t);
}

__global__ void rayTraceKernel(uchar4* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float u = float(x) / float(width);
    float v = float(y) / float(height);

    float3 origin = make_float3(0.0f, 0.0f, 0.0f);
    float3 direction = normalize(make_float3(u - 0.5f, v - 0.5f, -1.0f));

    float3 color = traceRay(origin, direction);

    output[idx] = make_uchar4(color.x * 255, color.y * 255, color.z * 255, 255);
}

// Wrapper function to launch the kernel
void launchRayTraceKernel(uchar4 * d_output, int width, int height) {
    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    rayTraceKernel<<<gridSize, blockSize>>>(d_output, width, height);

    // Ensure kernel launch is successful
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch rayTraceKernel (error code %s)!\n", cudaGetErrorString(err));
    }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
}