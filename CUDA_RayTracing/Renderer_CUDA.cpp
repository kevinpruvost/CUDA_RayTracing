#include "Renderer.h"
#include "Raytracer.cuh"

// Renderer function to launch the kernel and work with surfaces
void Renderer::launchCudaKernel(cudaArray* textureArray, int width, int height)
{
    // Create a surface object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = textureArray;

    cudaSurfaceObject_t surfaceObject = 0;
    cudaCreateSurfaceObject(&surfaceObject, &resDesc);

    // Launch the kernel via the wrapper function
    launchRayTraceKernel(surfaceObject, width, height);

    // Clean up
    cudaDestroySurfaceObject(surfaceObject);
}