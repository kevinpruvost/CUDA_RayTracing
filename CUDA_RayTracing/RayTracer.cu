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

__device__ __shared__ curandState globalState[N_BLOCK];

__global__ void initCurand(unsigned long * seed)
{
    int idx = threadIdx.x;
    curand_init(seed[idx], idx, 0, &globalState[idx]);
}

__device__ float frand()
{
    int idx = threadIdx.x;
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

    if (x >= width || y >= height) {
        return;
    }

    double3 color = make_double3(0.0f, 0.0f, 0.0f);
    for (int i = -RESAMPLING_SIZE / 2; i < RESAMPLING_SIZE / 2; ++i) {
        for (int j = -RESAMPLING_SIZE / 2; j < RESAMPLING_SIZE / 2; ++j) {
            double u = double(x) + ((double)i / (RESAMPLING_SIZE));
            double v = double(y) + ((double)j / (RESAMPLING_SIZE));

            Cuda_Camera* c = &scene->camera;
            double3 origin = c->O;
            // N + Dy * (2 * i / H - 1) + Dx * (2 * j / W - 1)
            double3 direction = c->N + c->Dy * (2 * (double)v / c->H - 1) + c->Dx * (2 * (double)u / c->W - 1);

            color += traceRay(scene, origin, direction, 1) / ((RESAMPLING_SIZE) * (RESAMPLING_SIZE));
        }
    }
    uchar4 outputColor;
    outputColor = make_uchar4(color.x * 255, color.y * 255, color.z * 255, 255);

    surf2Dwrite(outputColor, surface, x * sizeof(uchar4), y);
}

// Wrapper function to launch the kernel
void launchRayTraceKernel(cudaSurfaceObject_t surface, int width, int height, Cuda_Scene * scene)
{
    // Define block and grid sizes
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    unsigned long* seeds;
    cudaMemcpy(&seeds, &scene->seeds, sizeof(unsigned long*), cudaMemcpyDeviceToHost);
    initCurand << <blocksPerGrid, threadsPerBlock >> > (seeds);
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