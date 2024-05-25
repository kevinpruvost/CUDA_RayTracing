#include "Primitive.cuh"

__device__ bool IntersectBoundingBox(const double3* origin, const double3* direction, const double3* min, const double3* max)
{
    double3 invDirection = make_double3(1.0 / direction->x, 1.0 / direction->y, 1.0 / direction->z);

    double3 t0 = (*min - *origin) * invDirection;
    double3 t1 = (*max - *origin) * invDirection;

    double3 tmin = make_double3(fmin(t0.x, t1.x), fmin(t0.y, t1.y), fmin(t0.z, t1.z));
    double3 tmax = make_double3(fmax(t0.x, t1.x), fmax(t0.y, t1.y), fmax(t0.z, t1.z));

    double tminmax = fmax(fmax(tmin.x, tmin.y), tmin.z);
    double tmaxmin = fmin(fmin(tmax.x, tmax.y), tmax.z);

    return tminmax <= tmaxmin;
}

__device__ bool IntersectTriangle(const double3* origin, const double3* direction, const Cuda_Triangle * triangle, Cuda_Collision & tmp)
{
    if (triangle == nullptr) return false;
    const double EPSILON = 1e-6;

    double3 edge1 = triangle->O2 - triangle->O1;
    double3 edge2 = triangle->O3 - triangle->O1;

    double3 p = cross(*direction, edge2);
    double det = dot(edge1, p);

    if (fabs(det) < EPSILON) return false;

    double invDet = 1.0 / det;
    double3 t = *origin - triangle->O1;

    double u = dot(t, p) * invDet;
    if (u < 0.0 || u > 1.0) return false;

    double3 q = cross(t, edge1);
    double v = dot(*direction, q) * invDet;
    if (v < 0.0 || u + v > 1.0) return false;

    double t0 = dot(edge2, q) * invDet;
    if (t0 <= EPSILON) return false;

    tmp.isCollide = true;
    tmp.C = *origin + *direction * t0;

    double3 triangleNormal = triangle->N;
    
    // Determine if the intersection is in front or behind the triangle based on the normal and the direction
    if (dot(*direction, triangleNormal) < 0) {
        tmp.front = true;
        tmp.N = triangleNormal;
    }
    else {
        tmp.front = false;
        tmp.N = -triangleNormal;
    }

    tmp.dist = t0;

    return true;
}

__device__ bool MeshIntersect(Cuda_Mesh * mesh, const double3* origin, const double3* direction, Cuda_Collision* collision, int depth)
{
    if (mesh == nullptr) return false;
    Cuda_Collision tmp = InitCudaCollision();
    if (mesh->left == nullptr && mesh->right == nullptr)
    {
        if (IntersectTriangle(origin, direction, mesh->triangle, tmp) && tmp.dist < collision->dist)
            *collision = tmp;
        return collision->isCollide;
    }
    if (!IntersectBoundingBox(origin, direction, &mesh->min, &mesh->max)) return false;
    bool f1 = MeshIntersect(mesh->left, origin, direction, collision, depth + 1);
    bool f2 = MeshIntersect(mesh->right, origin, direction, collision, depth + 1);
    return f1 || f2;
}

__device__ void MeshIntersect(Cuda_Primitive* primitive, const double3* origin, const double3* direction, Cuda_Collision* collision)
{
    Cuda_Mesh* mesh = &primitive->data.mesh;
    if (MeshIntersect(mesh, origin, direction, collision, 0))
    {
        collision->collide_primitive = primitive;
    }
}