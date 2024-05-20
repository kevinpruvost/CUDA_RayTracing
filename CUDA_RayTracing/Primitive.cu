#include "Primitive.cuh"
#include <iostream>

__device__ Cuda_Collision InitCudaCollision()
{
    Cuda_Collision collision;
    collision.isCollide = false;
    collision.collide_primitive = NULL;
    collision.dist = BIG_DIST;
    return collision;
}

__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 operator*(float b, const float3& a) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ uchar3 operator*(const uchar3& a, float b) {
    return make_uchar3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 operator+=(float3& a, const uchar3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__device__ float3 operator/=(const float3& a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ float3 GetMaterialSmoothPixel(Cuda_Material* material, float u, float v)
{
    const float EPS = 1e-6;

    // Calculate the positions in the texture
    float U = (u - floorf(u)) * material->texture_height;
    float V = (v - floorf(v)) * material->texture_width;
    int U1 = static_cast<int>(floorf(U - EPS));
    int U2 = U1 + 1;
    int V1 = static_cast<int>(floorf(V - EPS));
    int V2 = V1 + 1;
    float rat_U = U2 - U;
    float rat_V = V2 - V;

    // Handle wrapping
    if (U1 < 0) U1 = material->texture_height - 1;
    if (U2 == material->texture_height) U2 = 0;
    if (V1 < 0) V1 = material->texture_width - 1;
    if (V2 == material->texture_width) V2 = 0;

    // Perform bilinear interpolation
    float3 color = make_float3(0, 0, 0);

    color += material->texture[U1 * material->texture_height + V1] * rat_U * rat_V;
    color += material->texture[U1 * material->texture_height + V2] * rat_U * (1 - rat_V);
    color += material->texture[U2 * material->texture_height + V1] * (1 - rat_U) * rat_V;
    color += material->texture[U2 * material->texture_height + V2] * (1 - rat_U) * (1 - rat_V);
    color /= 256.0f;

    return color;
}

__device__ bool intersect(Cuda_Primitive* primitive, float3 origin, float3 direction, Cuda_Collision* collision)
{
    return false;
}
