#include "Scene.cuh"

__device__ Cuda_Primitive* intersect(Cuda_Scene* scene, float3 origin, float3 direction, Cuda_Collision* collision)
{
    Cuda_Primitive * collide_primitive = nullptr;
    collision->dist = BIG_DIST;

    for (int i = 0; i < scene->primitiveCount; i++)
    {
        Cuda_Primitive * primitive = &scene->primitives[i];
        Cuda_Collision temp_collision;
        if (intersect(primitive, origin, direction, &temp_collision))
        {
            if (temp_collision.dist < collision->dist)
            {
                *collision = temp_collision;
                collide_primitive = primitive;
            }
        }
    }

    return collide_primitive;
}

__device__ float3 traceRay(Cuda_Scene* scene, double u, double v)
{
    float3 color;

    Cuda_Camera* c;
    float3 origin = scene->camera.O;
    // N + Dy * (2 * i / H - 1) + Dx * (2 * j / W - 1)
    float3 direction = c->N + c->Dy * (2 * u / c->H - 1) + c->Dx * (2 * v / c->W - 1);

    return color;
}