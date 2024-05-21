#include "Scene.cuh"

__device__ Cuda_Collision intersect(Cuda_Scene* scene, const double3 * origin, const double3 * direction)
{
    Cuda_Collision collision = InitCudaCollision();
    Cuda_Collision empty = collision;

    for (int i = 0; i < scene->primitiveCount; i++)
    {
        Cuda_Primitive* primitive = &scene->primitives[i];
        Cuda_Collision temp_collision = empty;
        if (intersect(primitive, origin, direction, &temp_collision))
        {
            if (temp_collision.dist - collision.dist < 1e-6)
            {
                collision = temp_collision;
            }
        }
    }
    return collision;
}

__device__ const int max_depth = 10;
__device__ double3 traceRay(Cuda_Scene* scene, double3 origin, double3 direction, int depth)
{
    double3 color = make_double3(0.0f, 0.0f, 0.0f);

    if (depth > max_depth) return color;

    Cuda_Collision collision;
    collision = intersect(scene, &origin, &direction);
    if (collision.isCollide)
    {
        Cuda_Primitive * prim = collision.collide_primitive;
        if (prim->isLightPrimitive)
            color = collision.collide_primitive->material.color;
        else {
            if (prim->material.diff > 1e-6 || prim->material.spec > 1e-6) {
                color += CalnDiffusion(scene, &collision);
            }
            if (prim->material.refl > 1e-6) {
                color += CalnReflection(scene, &collision, direction, depth);
            }
            if (prim->material.refr > 1e-6) {
                color += CalnRefraction(scene, &collision, direction, depth);
            }
        }
    }
    else
    {
        double3 c = normalize(direction);
        double t = 0.5 * (c.y + 1.0);
        color = scene->backgroundColor_bottom * (1.0 - t) + scene->backgroundColor_top * t;
    }
    color.x = fmin(1.0, color.x);
    color.y = fmin(1.0, color.y);
    color.z = fmin(1.0, color.z);
    return color;
}