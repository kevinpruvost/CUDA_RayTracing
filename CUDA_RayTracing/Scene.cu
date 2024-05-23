#include "Scene.cuh"

__device__ bool intersectAABB(const double3& origin, const double3& direction, const double3& aabb_min, const double3& aabb_max) {
    double3 invDir = 1.0 / direction;

    double3 t0s = (aabb_min - origin) * invDir;
    double3 t1s = (aabb_max - origin) * invDir;

    double3 tsmaller = fmin(t0s, t1s);
    double3 tbigger = fmax(t0s, t1s);

    double tmin = fmax(fmax(tsmaller.x, tsmaller.y), tsmaller.z);
    double tmax = fmin(fmin(tbigger.x, tbigger.y), tbigger.z);

    return tmax >= tmin;
}

__device__ bool traverseBVH(const double3 * origin, const double3 * direction, const Cuda_BVH * node, Cuda_Collision * collision, Cuda_Primitive ** ignorePrimitive)
{
    if (node == nullptr) return false;
    if (!intersectAABB(*origin, *direction, node->min, node->max)) return false;

    bool ignore = false;
    for (int i = 0; i < 2 && ignorePrimitive != nullptr; i++)
    {
        if (ignorePrimitive[i] == node->primitive)
        {
            ignore = true;
            break;
        }
    }

    if (node->primitive != nullptr || ignore)
    {
        Cuda_Collision temp_collision = InitCudaCollision();
        if (intersect(node->primitive, origin, direction, &temp_collision))
        {
            if (temp_collision.dist < collision->dist)
                *collision = temp_collision;
            return true;
        }
    }
    else
    {
        bool hit_left = traverseBVH(origin, direction, node->left, collision, ignorePrimitive);
        bool hit_right = traverseBVH(origin, direction, node->right, collision, ignorePrimitive);
        return hit_left || hit_right;
    }
    return false;
}

__device__ Cuda_Collision intersect(Cuda_BVH * bvh, const double3* origin, const double3* direction, Cuda_Primitive ** ignorePrimitive)
{
    Cuda_Collision collision = InitCudaCollision();
    Cuda_Collision empty = collision;

    traverseBVH(origin, direction, bvh, &collision, ignorePrimitive);

    return collision;
}

//__device__ Cuda_Collision intersect(Cuda_Scene* scene, const double3 * origin, const double3 * direction)
//{
//    Cuda_Collision collision = InitCudaCollision();
//    Cuda_Collision empty = collision;
//
//    for (int i = 0; i < scene->primitiveCount; i++)
//    {
//        Cuda_Primitive* primitive = &scene->primitives[i];
//        Cuda_Collision temp_collision = empty;
//        if (intersect(primitive, origin, direction, &temp_collision))
//        {
//            if (temp_collision.dist - collision.dist < 1e-6)
//            {
//                collision = temp_collision;
//            }
//        }
//    }
//    return collision;
//}

__device__ const int max_depth = 10;

__device__ double3 traceRay(Cuda_Scene* scene, double3 origin, double3 direction, int depth)
{
    double3 color = make_double3(0.0f, 0.0f, 0.0f);

    if (depth > max_depth) return color;

    Cuda_Collision collision;
    collision = intersect(scene->bvh, &origin, &direction);
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