#include "Scene.cuh"

__device__ bool intersectAABB(const double3& origin, const double3& direction, const double3& aabb_min, const double3& aabb_max) {
    double tmin = (aabb_min.x - origin.x) / direction.x;
    double tmax = (aabb_max.x - origin.x) / direction.x;

    if (tmin > tmax) {
        double temp = tmin;
        tmin = tmax;
        tmax = temp;
    }

    double tymin = (aabb_min.y - origin.y) / direction.y;
    double tymax = (aabb_max.y - origin.y) / direction.y;

    if (tymin > tymax) {
        double temp = tymin;
        tymin = tymax;
        tymax = temp;
    }

    if ((tmin > tymax) || (tymin > tmax)) {
        return false;
    }

    if (tymin > tmin) {
        tmin = tymin;
    }

    if (tymax < tmax) {
        tmax = tymax;
    }

    double tzmin = (aabb_min.z - origin.z) / direction.z;
    double tzmax = (aabb_max.z - origin.z) / direction.z;

    if (tzmin > tzmax) {
        double temp = tzmin;
        tzmin = tzmax;
        tzmax = temp;
    }

    if ((tmin > tzmax) || (tzmin > tmax)) {
        return false;
    }

    if (tzmin > tmin) {
        tmin = tzmin;
    }

    if (tzmax < tmax) {
        tmax = tzmax;
    }

    return true;
}

__device__ bool traverseBVH(const double3 * origin, const double3 * direction, const Cuda_BVH * node, Cuda_Collision * collision, Cuda_Primitive * ignorePrimitive, Cuda_Primitive * ignorePrimitive2)
{
    return false;
    if (node == nullptr) return false;
    if (!intersectAABB(*origin, *direction, node->min, node->max))
        return false;

    bool ignore = node->primitive == 0 || ignorePrimitive == node->primitive || ignorePrimitive2 == node->primitive;
    if (ignore)
    {
        Cuda_Collision temp_collision = InitCudaCollision();
        if (intersect_primitives(node->primitive, origin, direction, &temp_collision))
        {
            if (temp_collision.dist < collision->dist)
                *collision = temp_collision;
            return true;
        }
    }
    else
    {
        bool hit_left = traverseBVH(origin, direction, node->left, collision, ignorePrimitive, ignorePrimitive2);
        bool hit_right = traverseBVH(origin, direction, node->right, collision, ignorePrimitive, ignorePrimitive2);
        return hit_left || hit_right;
    }
    return false;
}

__device__ Cuda_Collision intersect(Cuda_BVH * bvh, const double3* origin, const double3* direction, Cuda_Primitive * ignorePrimitive, Cuda_Primitive * ignorePrimitive2)
{
    Cuda_Collision collision = InitCudaCollision();
    traverseBVH(origin, direction, bvh, &collision, ignorePrimitive, ignorePrimitive2);
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

__device__ double3 traceRay(Cuda_Scene* scene, const double3 & origin, const double3 & direction, int depth)
{
    double3 color = make_double3(0.0f, 0.0f, 0.0f);

    if (depth > max_depth) return color;

    Cuda_Collision collision = InitCudaCollision();
    collision = intersect(scene->bvh, &origin, &direction, 0, 0);
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