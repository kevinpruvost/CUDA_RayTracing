#include <cuda_runtime.h>
#include <math_functions.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "RayTracer.cuh"

__device__ __shared__ curandState globalState[N_BLOCK];

__global__ void initCurand(unsigned long * seed)
{
    int idx = threadIdx.x;
    curand_init(idx, idx, 0, &globalState[idx]);
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

__device__ double2 randomInUnitDisk() {
    double2 p;
    do {
        p = 2.0 * make_double2(frand(), frand()) - make_double2(1.0, 1.0);
    } while (dot(p, p) >= 0.5);
    return p;
}

__device__ void modifyRayForDepthOfField(double3& origin, double3& direction, double focalDistance, double aperture, Cuda_Camera * camera, const double3 & focal_point)
{
    double2 random = make_double2(frand() - 0.5, frand() - 0.5) * aperture / 10.0f;
    double3 offset = camera->Dx * random.x + camera->Dy * random.y;

    origin += offset;
    direction = normalize(focal_point - origin);
}

__global__ void rayTraceKernel(cudaSurfaceObject_t surface, int texture_width, int texture_height, int viewport_width, int viewport_height, Cuda_Scene* scene, int x_progress, int y_progress, int width_to_do, int height_to_do, Settings * settings)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width_to_do || y >= height_to_do) {
        return;
    }

    x += x_progress;
    y += y_progress;

    int surface_x = x;
    int surface_y = y;

    x = (double)x / texture_width * scene->camera.W;
    y = (double)y / texture_height * scene->camera.H;

    double3 color = make_double3(0.0f, 0.0f, 0.0f);
    double3 origin = scene->camera.O;
    double3 d = normalize(scene->camera.N + scene->camera.Dy * (2 * (double)y / scene->camera.H - 1) + scene->camera.Dx * (2 * (double)x / scene->camera.W - 1));
    double3 focal_point = origin + settings->depthOfField.focalDistance * d;

    if (settings->resampling_size == 1)
    {
        double3 direction = normalize(scene->camera.N + scene->camera.Dy * (2 * (double)y / scene->camera.H - 1) + scene->camera.Dx * (2 * (double)x / scene->camera.W - 1));
        if (settings->depthOfField.enabled) {
            modifyRayForDepthOfField(origin, direction, settings->depthOfField.focalDistance, settings->depthOfField.aperture, &scene->camera, focal_point);
        }
        color = traceRay(scene, origin, direction, 1);
    }
    else {
        // Monte Carlo Ray Tracing
        Cuda_Camera* c = &scene->camera;
        for (int i = 0; i < settings->resampling_size; ++i) {
            double u = x + (2.0f * frand() - 1.0f);
            double v = y + (2.0f * frand() - 1.0f);
            double3 originD = c->O;
            double3 direction = normalize(c->N + c->Dy * (2 * (double)v / c->H - 1) + c->Dx * (2 * (double)u / c->W - 1));
            if (settings->depthOfField.enabled) {
                modifyRayForDepthOfField(originD, direction, settings->depthOfField.focalDistance, settings->depthOfField.aperture, &scene->camera, focal_point);
            }
            color += traceRay(scene, originD, direction, 1);
        }
        color /= settings->resampling_size;
    }
    uchar4 outputColor;
    outputColor = make_uchar4(color.x * 255, color.y * 255, color.z * 255, 255);

    surf2Dwrite(outputColor, surface, surface_x * sizeof(uchar4), surface_y);
}

// Wrapper function to launch the kernel
void launchRayTraceKernel(cudaSurfaceObject_t surface, int texture_width, int texture_height, int viewport_width, int viewport_height, Cuda_Scene * scene, int x_progress, int y_progress, int width_to_do, int height_to_do, Settings * settings)
{
    // Define block and grid sizes
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((width_to_do + threadsPerBlock.x - 1) / threadsPerBlock.x, (height_to_do + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    unsigned long* seeds;
    cudaMemcpy(&seeds, &scene->seeds, sizeof(unsigned long*), cudaMemcpyDeviceToHost);
    initCurand << <blocksPerGrid, threadsPerBlock >> > (seeds);
    cudaDeviceSynchronize();
    rayTraceKernel<<<blocksPerGrid, threadsPerBlock >> > (surface, texture_width, texture_height, viewport_width, viewport_height, scene, x_progress, y_progress, width_to_do, height_to_do, settings);

    // Ensure kernel launch is successful
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch rayTraceKernel (error code %s)!\n", cudaGetErrorString(err));
    }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
}