#include "Primitive.cuh"
#include <iostream>

__device__ Cuda_Collision InitCudaCollision()
{
    Cuda_Collision collision;
    collision.isCollide = false;
    collision.collide_primitive = NULL;
    collision.dist = BIG_DIST;
    collision.front = false;
    collision.N = make_double3(0, 0, 0);
    collision.C = make_double3(0, 0, 0);
    return collision;
}

__device__ double3 GetTextureColor(Cuda_Collision* collision)
{
    double3 color;
    switch (collision->collide_primitive->type)
    {
        case Cuda_Primitive_Type_Sphere:
        {
            Cuda_Sphere* sphere = &collision->collide_primitive->data.sphere;

            double3 I = normalize(collision->C - sphere->O);
            double a = acos(-dot(I, sphere->De));
            double b = acos(fmin(fmax(dot(I, sphere->Dc) / sinf(a), -1.0), 1.0));
            double u = a / M_PI, v = b / (2 * M_PI);
            if (dot(I, cross(sphere->Dc, sphere->De)) < 0) v = 1 - v;
            color = GetMaterialSmoothPixel(&collision->collide_primitive->material, u, v);
            break;
        }
        case Cuda_Primitive_Type_Plane:
        {
            Cuda_Plane* plane = &collision->collide_primitive->data.plane;

            double u = dot(collision->C, plane->Dx) / dot(plane->Dx, plane->Dx);
            double v = dot(collision->C, plane->Dy) / dot(plane->Dy, plane->Dy);
            color = GetMaterialSmoothPixel(&collision->collide_primitive->material, u, v);
            break;
        }
        case Cuda_Primitive_Type_Square:
        {
            Cuda_Square* square = &collision->collide_primitive->data.square;

            double u = dot(collision->C, square->Dx) / dot(square->Dx, square->Dx) / 2.0 + 0.5;
            double v = dot(collision->C, square->Dy) / dot(square->Dy, square->Dy) / 2.0 + 0.5;
            color = GetMaterialSmoothPixel(&collision->collide_primitive->material, u, v);
            break;        
        }
        case Cuda_Primitive_Type_Cylinder:
        {
            Cuda_Cylinder* cylinder = &collision->collide_primitive->data.cylinder;

            double u, v;
            if (!collision->front)
            {
                // Body
                double3 circle_point = collision->C - cylinder->O1;
                double angle = atan2(circle_point.z, circle_point.x);
                if (angle < 0) angle += 2 * M_PI;

                u = angle / (2 * M_PI);
                v = dot(collision->C - cylinder->O1, normalize(cylinder->O2 - cylinder->O1)) / length(cylinder->O2 - cylinder->O1);
            }
            else
            {
                // Caps
                // Calculate position of Collision point on cap circle
                double3 C = collision->C;
                double3 cap_center = length(C - cylinder->O1) < length(C - cylinder->O2) ? cylinder->O1 : cylinder->O2;
                double3 local_coords = C - cap_center;

                u = (local_coords.x / (2.0 * cylinder->R) + 0.5);
                v = (local_coords.z / (2.0 * cylinder->R) + 0.5);
            }

            color = GetMaterialSmoothPixel(&collision->collide_primitive->material, u, v);
            break;
        }
    }
    return color;
}

__device__ double3 GetMaterialSmoothPixel(Cuda_Material* material, double u, double v)
{
    const double EPS = 1e-6;

    // Calculate the positions in the texture
    double U = (u - floorf(u)) * material->texture_height;
    double V = (v - floorf(v)) * material->texture_width;
    int U1 = static_cast<int>(floorf(U - EPS));
    int U2 = U1 + 1;
    int V1 = static_cast<int>(floorf(V - EPS));
    int V2 = V1 + 1;
    double rat_U = U2 - U;
    double rat_V = V2 - V;

    // Handle wrapping
    if (U1 < 0) U1 = material->texture_height - 1;
    if (U2 == material->texture_height) U2 = 0;
    if (V1 < 0) V1 = material->texture_width - 1;
    if (V2 == material->texture_width) V2 = 0;

    // Perform bilinear interpolation
    double3 color = make_double3(0, 0, 0);

    color += material->texture[V1 * material->texture_height + U1] * rat_U * rat_V;
    color += material->texture[V1 * material->texture_height + U2] * rat_U * (1 - rat_V);
    color += material->texture[V2 * material->texture_height + U1] * (1 - rat_U) * rat_V;
    color += material->texture[V2 * material->texture_height + U2] * (1 - rat_U) * (1 - rat_V);
    color /= 256.0f;

    return color;
}