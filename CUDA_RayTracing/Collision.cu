#include "Primitive.cuh"

__device__ void CylinderIntersect(Cuda_Primitive* primitive, const double3* origin, const double3* direction, Cuda_Collision* collision)
{
    double3 V = normalize(*direction);
    // Body
    {
        // Cylinder collision detection
        double3 d = primitive->data.cylinder.O2 - primitive->data.cylinder.O1;
        double3 m = *origin - primitive->data.cylinder.O1;
        double3 n = *origin - primitive->data.cylinder.O2;
        double md = dot(m, d);
        double nd = dot(n, d);
        double dd = dot(d, d);
        double r2 = primitive->data.cylinder.R * primitive->data.cylinder.R;

        // Cylinder body collision detection
        double3 dNorm = normalize(d);
        //double3 mProj = m - dot(m, dNorm) * dNorm;
        //double3 VProj = V - dot(V, dNorm) * dNorm;
        double3 mProj = cross(m, dNorm);
        double3 VProj = cross(V, dNorm);
        double a = dot(VProj, VProj);
        double b = 2.0 * dot(VProj, mProj);
        double c = dot(mProj, mProj) - r2;
        double det = b * b - 4 * a * c;

        if (det >= 0)
        {
            det = sqrt(det);
            double t0 = (-b - det) / (2.0 * a);
            double t1 = (-b + det) / (2.0 * a);

            double t = fmin(t0, t1);
            if (t < 0) t = fmax(t0, t1);
            if (t >= 0)
            {
                double3 P = *origin + V * t;
                //double3 Q = P - primitive->data.cylinder.O1;
                double u = dot(P - primitive->data.cylinder.O1, dNorm);

                if (u >= 0 && u <= length(d))
                {
                    // Indicate body
                    collision->dist = t;
                    collision->C = P;
                    collision->isCollide = true;
                    collision->collide_primitive = primitive;
                    collision->front = false;
                    collision->N = normalize(P - (primitive->data.cylinder.O1 + dNorm * u));
                    return;
                }
                else
                {
                }
            }
        }
    }
    // Top Cap
    {
        double3 N = normalize(primitive->data.cylinder.O1 - primitive->data.cylinder.O2);
        double denom = dot(N, V);
        if (fabs(denom) >= 1e-6)
        {
            float t = dot(primitive->data.cylinder.O2 - *origin, N) / denom;
            if (t >= 1e-6)
            {
                double3 P = *origin + V * t;
                double3 Q = P - primitive->data.cylinder.O2;
                if (dot(Q, Q) <= primitive->data.cylinder.R * primitive->data.cylinder.R)
                {
                    collision->dist = t;
                    collision->C = P;
                    collision->front = true;
                    collision->N = (denom < 0) ? N : -N;
                    collision->isCollide = true;
                    collision->collide_primitive = primitive;
                    return;
                }
            }
        }
    }
    // Bottom cap
    {
        double3 N = normalize(primitive->data.cylinder.O2 - primitive->data.cylinder.O1);
        double denom = dot(N, V);
        if (fabs(denom) >= 1e-6)
        {
            float t = dot(primitive->data.cylinder.O1 - *origin, N) / denom;
            if (t >= 1e-6)
            {
                double3 P = *origin + V * t;
                double3 Q = P - primitive->data.cylinder.O1;
                if (dot(Q, Q) <= primitive->data.cylinder.R * primitive->data.cylinder.R)
                {
                    collision->dist = t;
                    collision->C = P;
                    collision->front = true;
                    collision->N = (denom < 0) ? N : -N;
                    collision->isCollide = true;
                    collision->collide_primitive = primitive;
                    return;
                }
            }
        }
    }
}

// Function to check triangle-ray intersection using Möller–Trumbore algorithm
__device__ bool TriangleIntersect(const Cuda_Triangle* triangle, const double3* origin, const double3* direction, double* t, double3* P, double3* N)
{
    const double EPSILON = 1e-6;
    double3 V1 = triangle->O1;
    double3 V2 = triangle->O2;
    double3 V3 = triangle->O3;

    double3 edge1 = V2 - V1;
    double3 edge2 = V3 - V1;
    double3 h = cross(*direction, edge2);
    double a = dot(edge1, h);

    if (fabs(a) < EPSILON)
        return false; // This ray is parallel to this triangle.

    double f = 1.0 / a;
    double3 s = *origin - V1;
    double u = f * dot(s, h);

    if (u < 0.0 || u > 1.0)
        return false;

    double3 q = cross(s, edge1);
    double v = f * dot(*direction, q);

    if (v < 0.0 || u + v > 1.0)
        return false;

    double temp = f * dot(edge2, q);

    if (temp > EPSILON) // ray intersection
    {
        *t = temp;
        *P = *origin + (*direction) * (*t);
        *N = normalize(cross(edge1, edge2));
        return true;
    }
    else // This means that there is a line intersection but not a ray intersection.
        return false;
}

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
            //ret.O3 = P;
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
        case Cuda_Primitive_Type_Cylinder:
        {
            CylinderIntersect(primitive, origin, direction, collision);
            break;
        }
        case Cuda_Primitive_Type_Bezier:
        {
            BezierIntersect(primitive, origin, direction, collision);
            break;
        }
        case Cuda_Primitive_Type_Triangle:
        {
            double t;
            double3 P, N;
            if (TriangleIntersect(&(primitive->data.triangle), origin, direction, &t, &P, &N))
            {
                collision->dist = t;
                collision->C = P;
                collision->N = N;
                collision->isCollide = true;
                collision->collide_primitive = primitive;
                collision->front = dot(*direction, N) < 0;
                return true;
            }
            break;
        }
        case Cuda_Primitive_Type_Mesh:
        {
            MeshIntersect(primitive, origin, direction, collision);
            break;
        }
    }

    return collision->isCollide;
}
