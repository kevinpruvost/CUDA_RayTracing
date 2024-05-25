#include <cuda_runtime.h>
#include <math_functions.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <thread>
#include <atomic>
#include "RayTracer.cuh"

__device__ __shared__ curandState globalState[N_BLOCK];

__global__ void initCurand(unsigned long* seed)
{
    int idx = threadIdx.x;
    curand_init(idx, idx, 0, &globalState[idx]);
}

__device__ float frand()
{
    int idx = threadIdx.x;
    return curand_uniform(&globalState[idx]);
}

__device__ double2 randomInUnitDisk() {
    double2 p;
    do {
        p = 2.0 * make_double2(frand(), frand()) - make_double2(1.0, 1.0);
    } while (dot(p, p) >= 0.5);
    return p;
}

__device__ void modifyRayForDepthOfField(double3& origin, double3& direction, double focalDistance, double aperture, Cuda_Camera* camera, const double3& focal_point)
{
    double2 random = make_double2(frand() - 0.5, frand() - 0.5) * aperture / 10.0f;
    double3 offset = camera->Dx * random.x + camera->Dy * random.y;

    origin += offset;
    direction = normalize(focal_point - origin);
}

__global__ void rayTraceKernel(cudaSurfaceObject_t surface, int texture_width, int texture_height, int viewport_width, int viewport_height, Cuda_Scene* scene, int x_progress, int y_progress, int width_to_do, int height_to_do, Settings* settings, int* progress)
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

    if (progress) {
        atomicAdd(progress, 1); // Atomically update progress
    }
}

// Wrapper function to launch the kernel
void launchRayTraceKernel(cudaSurfaceObject_t surface, int texture_width, int texture_height, int viewport_width, int viewport_height, Cuda_Scene* scene, int x_progress, int y_progress, int width_to_do, int height_to_do, Settings* settings)
{
    // Define block and grid sizes
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((width_to_do + threadsPerBlock.x - 1) / threadsPerBlock.x, (height_to_do + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    unsigned long* seeds;
    cudaMemcpy(&seeds, &scene->seeds, sizeof(unsigned long*), cudaMemcpyDeviceToHost);
    initCurand << <blocksPerGrid, threadsPerBlock >> > (seeds);
    cudaDeviceSynchronize();
    rayTraceKernel << <blocksPerGrid, threadsPerBlock >> > (surface, texture_width, texture_height, viewport_width, viewport_height, scene, x_progress, y_progress, width_to_do, height_to_do, settings, nullptr);

    // Ensure kernel launch is successful
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch rayTraceKernel (error code %s)!\n", cudaGetErrorString(err));
    }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
}

void printProgress(int textureWidth, int textureHeight, int* progress, bool* done)
{
    int seconds = 0;
    while (!*done)
    {
        std::cout << "Progress: " << *progress << " " << (double)(*progress) / (double)(textureWidth * textureHeight) * 100 << "%" << std::endl;
        std::cout << "Time remaining: " << (double)(textureWidth * textureHeight - *progress) / (double)(*progress + 1) * seconds << "s" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        seconds++;
    }
    std::cout << "Completed !" << std::endl;
}

// Wrapper function to launch the kernel
void launchRayTraceKernelParallel(cudaSurfaceObject_t surface, int texture_width, int texture_height, int viewport_width, int viewport_height, Cuda_Scene* scene, Settings* settings)
{
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    // Define block and grid sizes
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((texture_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (texture_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    unsigned long* seeds;
    cudaMemcpy(&seeds, &scene->seeds, sizeof(unsigned long*), cudaMemcpyDeviceToHost);
    initCurand << <blocksPerGrid, threadsPerBlock >> > (seeds);
    cudaDeviceSynchronize();

    bool done = false;
    int* progress;
    cudaMallocHost(&progress, sizeof(int)); // Allocate page-locked memory for progress
    *progress = 0;

    std::thread progressThread(printProgress, texture_width, texture_height, progress, &done);

    rayTraceKernel << <blocksPerGrid, threadsPerBlock >> > (surface, texture_width, texture_height, viewport_width, viewport_height, scene, 0, 0, texture_width, texture_height, settings, progress);

    // Ensure kernel launch is successful
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch rayTraceKernel (error code %s)!\n", cudaGetErrorString(err));
    }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    done = true;
    progressThread.join();

    cudaFreeHost(progress); // Free the page-locked memory
}
