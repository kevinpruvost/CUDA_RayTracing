#include <cuda_runtime.h>
#include <math_functions.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "RayTracer.cuh"

// Utility function to normalize a double3
__device__ double3 normalize(const double3& v) {
    float len = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return {
        v.x / len,
        v.y / len,
        v.z / len
    };
}

__device__ curandState globalState[N_BLOCK];

__global__ void initCurand(unsigned long seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N_BLOCK) {
        curand_init(seed, idx, 0, &globalState[idx]);
    }
}

__device__ float frand()
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N_BLOCK) idx = idx % N_BLOCK;
    return curand_uniform(&globalState[idx]);
}

//// Utility function to perform linear interpolation
//__device__ double3 lerp(const double3& a, const double3& b, float t) {
//    return {
//        a.x + t * (b.x - a.x),
//        a.y + t * (b.y - a.y),
//        a.z + t * (b.z - a.z)
//    };
//}
//
//// Simple background color function
//__device__ double3 traceRay(const double3& origin, const double3& direction) {
//    // Linearly interpolate between white and blue based on the y coordinate
//    float t = 0.5f * (direction.y + 1.0f);
//    return lerp({ 1.0f, 1.0f, 1.0f }, { 0.5f, 0.7f, 1.0f }, t);
//}

__global__ void rayTraceKernel(cudaSurfaceObject_t surface, int width, int height, Cuda_Scene* scene) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    double u = double(x) / double(width);
    double v = double(y) / double(height);

    //double3 origin = make_double3(0.0f, 0.0f, 0.0f);
    //double3 direction = normalize(make_double3(u - 0.5f, v - 0.5f, -1.0f));

    double3 color = traceRay(scene, u, v, x, y, 1);
//    double3 color = make_double3(1.0f, 0.0f, 0.0);
    uchar4 outputColor;
    outputColor = make_uchar4(color.x * 255, color.y * 255, color.z * 255, 255);
    //for (int i = 0; i < scene->primitiveCount; ++i) {
    //    if (scene->primitives[i].material.texture_width != 0) {
    //        uchar3* texture = scene->primitives[i].material.texture;
    //        int texture_x = static_cast<int>(u * scene->primitives[i].material.texture_width);
    //        int texture_y = static_cast<int>(v * scene->primitives[i].material.texture_height);
    //        outputColor = make_uchar4(
    //            texture[texture_x * scene->primitives[i].material.texture_height + texture_y].x,
    //            texture[texture_x * scene->primitives[i].material.texture_height + texture_y].y,
    //            texture[texture_x * scene->primitives[i].material.texture_height + texture_y].z,
    //            255);
    //    }
    //}

    surf2Dwrite(outputColor, surface, x * sizeof(uchar4), y);
}

#ifdef _DEBUG
__global__ void debugTest(Cuda_Scene* scene) {
    // Printing Camera information
    //printf("Camera Information:\n");
    //printf("Position: (%f, %f, %f)\n", scene->camera.O.x, scene->camera.O.y, scene->camera.O.z);
    //printf("Direction: (%f, %f, %f)\n", scene->camera.N.x, scene->camera.N.y, scene->camera.N.z);

    //// Printing Light information
    //printf("Light Information:\n");
    //for (int i = 0; i < scene->lightCount; i++) {
    //    printf("Light %d:\n", i);
    //    printf("Position: (%f, %f, %f)\n", scene->lights[i].O.x, scene->lights[i].O.y, scene->lights[i].O.z);
    //    printf("Color: (%f, %f, %f)\n", scene->lights[i].color.x, scene->lights[i].color.y, scene->lights[i].color.z);
    //    printf("Type: %d\n", scene->lights[i].type);
    //}
    for (int i = 0; i < scene->primitiveCount; ++i) {
        if (scene->primitives[i].material.texture_width != 0) {
            uchar3 * texture = scene->primitives[i].material.texture;
            printf("Test read: Width:%d | height:%d | texture: %d,%d,%d\n", scene->primitives[i].material.texture_width, scene->primitives[i].material.texture_height, texture[0].x, texture[0].y, texture[0].z);
        }
    }
}
#endif

// Wrapper function to launch the kernel
void launchRayTraceKernel(cudaSurfaceObject_t surface, int width, int height, Cuda_Scene * scene)
{
    // Define block and grid sizes
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
#ifdef _DEBUG
    //debugTest << <1, 1 >> > (scene);
#endif
    initCurand<<<blocksPerGrid, threadsPerBlock>>>(time(0));
    cudaDeviceSynchronize();
    rayTraceKernel<<<blocksPerGrid, threadsPerBlock >> > (surface, width, height, scene);

    // Ensure kernel launch is successful
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch rayTraceKernel (error code %s)!\n", cudaGetErrorString(err));
    }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
}