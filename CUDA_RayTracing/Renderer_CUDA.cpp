#include "Renderer.h"
#include "Raytracer.cuh"

void Renderer::launchCudaKernel(cudaArray* textureArray, int width, int height)
{
    // Allocate device memory for the output image
    uchar4* d_output;
    cudaMalloc(&d_output, width * height * sizeof(uchar4));

    // Launch the kernel via the wrapper function
    launchRayTraceKernel(d_output, width, height);

    // Copy the results to the texture array
    cudaMemcpy2DToArray(textureArray, 0, 0, d_output, width * sizeof(uchar4), width * sizeof(uchar4), height, cudaMemcpyDeviceToDevice);

    // Cleanup
    cudaFree(d_output);
}