#include "Primitive.cuh"

__device__ bool intersect(Cuda_Primitive* primitive, double3 origin, double3 direction, Cuda_Collision* collision)
{
    switch (primitive->type)
    {
        case Cuda_Primitive_Type_Sphere:
        {
            double3 V = normalize(direction);
            double3 P = origin - primitive->data.sphere.O;
            double b = -dot(P, V);
            double det = b * b - dot(P, P) + primitive->data.sphere.R * primitive->data.sphere.R;

            if (det > 1e-6)
            {
                det = sqrt(det);
                double x1 = b - det, x2 = b + det;
                bool front = (x1 > 1e-6);
                double temp = (x1 > 1e-6) ? x1 : ((x2 > 1e-6) ? x2 : 0);
                if (temp < collision->dist && temp != 0)
                {
                    collision->dist = temp;
                    collision->C = origin + direction * collision->dist;
                    collision->front = front;
                    collision->N = normalize(collision->C - primitive->data.sphere.O);
                    if (collision->front == false) collision->N = -collision->N;
                    collision->isCollide = true;
                    collision->collide_primitive = primitive;
                }
            }
            break;
        }
        case Cuda_Primitive_Type_Plane:
        {
            direction = normalize(direction);
            double3 N = normalize(primitive->data.plane.N);
            double denom = dot(N, direction);
            if (fabs(denom) >= 1e-6)
            {
                float t = dot(N * primitive->data.plane.R - origin, N) / denom;
                if (t >= 1e-6 && t < collision->dist)
                {
                    collision->dist = t;
                    collision->C = origin + direction * collision->dist;
                    collision->front = (denom < 0);
                    collision->N = collision->front ? N : -N;
                    collision->isCollide = true;
                    collision->collide_primitive = primitive;
                }
            }
            break;
        }
    }

    return collision->isCollide;
}
