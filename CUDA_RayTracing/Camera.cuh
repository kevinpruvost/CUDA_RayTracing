#ifndef CAMERA_CUH
#define CAMERA_CUH

#include <cuda_runtime.h>
#define __CUDA_INTERNAL_COMPILATION__
#include <math_functions.h>
#undef __CUDA_INTERNAL_COMPILATION__

struct Cuda_Camera
{
    float3 O; // Position
    float3 N; // Normal
    float3 Dx; // Delta X, width camera
    float3 Dy; // Delta Y, height camera
    int W, H;
    double shade_quality;
    double drefl_quality;
    int max_photons;
    int emit_photons;
    int sample_photons;
    double sample_dist;
};

#endif