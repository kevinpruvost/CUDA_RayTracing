#include "Primitive.cuh"

__device__ bool intersect(Cuda_Primitive* primitive, const double3 * origin, const double3 * direction, Cuda_Collision* collision)
{
    switch (primitive->type)
    {
        case Cuda_Primitive_Type_Sphere:
        {
            double3 V = normalize(*direction);
            double3 P = *origin - primitive->data.sphere.O;
            double b = -dot(P, V);
            double det = b * b - dot(P, P) + primitive->data.sphere.R * primitive->data.sphere.R;

            if (det > 1e-6)
            {
                det = sqrt(det);
                double x1 = b - det, x2 = b + det;
                if (x2 < 1e-6) return false;
                collision->front = (x1 > 1e-6);
                collision->dist = collision->front ? x1 : x2;
                collision->C = *origin + (V * collision->dist);
                collision->N = normalize(collision->C - primitive->data.sphere.O);
                if (collision->front == false) collision->N = -collision->N;
                collision->isCollide = true;
                collision->collide_primitive = primitive;
            }
            break;
        }
        case Cuda_Primitive_Type_Plane:
        {
            double3 V = normalize(*direction);
            double3 N = normalize(primitive->data.plane.N);
            double denom = dot(N, V);
            if (fabs(denom) >= 1e-6)
            {
                float t = dot(N * primitive->data.plane.R - *origin, N) / denom;
                if (t >= 1e-6)
                {
                    collision->dist = t;
                    collision->C = *origin + V * collision->dist;
                    collision->front = (denom < 0);
                    collision->N = collision->front ? N : -N;
                    collision->isCollide = true;
                    collision->collide_primitive = primitive;
                }
            }
            break;
        }
        case Cuda_Primitive_Type_Square:
        {

            //ray_V = ray_V.GetUnitVector();
            //auto N = (Dx * Dy).GetUnitVector();
            //double d = N.Dot(ray_V);

            //if (fabs(d) < EPS) {
            //    return ret;
            //}

            //// solve equation
            //double t = (O - ray_O).Dot(N) / d;
            //if (t < EPS) {
            //    return ret;
            //}
            //auto P = ray_O + ray_V * t;

            //// check whether inside square
            //double DxLen2 = Dx.Module2();
            //double DyLen2 = Dy.Module2();

            //double x2 = abs((P - O).Dot(Dx));
            //double y2 = abs((P - O).Dot(Dy));
            //if (x2 > DxLen2 || y2 > DyLen2) {
            //    return ret;
            //}

            //ret.dist = t;
            //ret.front = (d < 0);
            //ret.C = P;
            //ret.N = (ret.front) ? N : -N;
            //ret.isCollide = true;
            //ret.collide_primitive = this;
            //return ret;
            
            double3 V = normalize(*direction);
            double3 N = normalize(cross(primitive->data.square.Dx, primitive->data.square.Dy));
            double denom = dot(N, V);
            if (fabs(denom) < 1e-6) return false;
            
            float t = dot(primitive->data.square.O - *origin, N) / denom;
            
            if (t < 1e-6) return false;

            double3 P = *origin + V * t;
            double DxLen2 = dot(primitive->data.square.Dx, primitive->data.square.Dx);
            double DyLen2 = dot(primitive->data.square.Dy, primitive->data.square.Dy);
            double x2 = fabs(dot(P - primitive->data.square.O, primitive->data.square.Dx));
            double y2 = fabs(dot(P - primitive->data.square.O, primitive->data.square.Dy));
            if (x2 > DxLen2 || y2 > DyLen2) return false;

            collision->dist = t;
            collision->front = (denom < 0);
            collision->C = P;
            collision->N = collision->front ? N : -N;
            collision->isCollide = true;
            collision->collide_primitive = primitive;
            break;
        }
    }

    return collision->isCollide;
}
