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
        double nn = dot(n, n);
        double mn = dot(m, n);
        double a = dd - md * md / dd;
        double b = dd - nd * nd / dd;
        double c = dd - mn * mn / dd;
        double e = dd - nn * nn / dd;
        double f = dd - md * nd / dd;
        double r2 = primitive->data.cylinder.R * primitive->data.cylinder.R;

        // Test for collision with infinite cylinder
        double a1 = dot(V, V) - pow(dot(V, d), 2) / dd;
        double b1 = dot(m, V) - md * dot(V, d) / dd;
        double c1 = dot(m, m) - pow(md, 2) / dd - r2;
        double det = b1 * b1 - a1 * c1;
        if (det >= 0)
        {
            det = sqrt(det);
            double t0 = (-b1 - det) / a1;
            double t1 = (-b1 + det) / a1;

            // Check for collision within cylinder caps
            if (t1 >= 1e-6)
            {
                if (t0 < 1e-6) t0 = t1;

                double3 P = *origin + V * t0;
                double3 Q = (P - primitive->data.cylinder.O1);
                double u = dot(Q, d) / dd;
                if (!(u < 0 || u > 1))
                {
                    collision->dist = t0;
                    collision->front = (dot(V, d) >= 0);
                    collision->C = P;
                    collision->N = normalize(P - (primitive->data.cylinder.O1 + u * d));
                    if (!collision->front) collision->N = -collision->N;
                    collision->isCollide = true;
                    collision->collide_primitive = primitive;
                    return;
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
                    collision->front = (denom < 0);
                    collision->N = collision->front ? -N : N;
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
                    collision->front = (denom < 0);
                    collision->N = collision->front ? -N : N;
                    collision->isCollide = true;
                    collision->collide_primitive = primitive;
                    return;
                }
            }
        }
    }
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
        case Cuda_Primitive_Type_Cylinder:
        {
            CylinderIntersect(primitive, origin, direction, collision);
            break;
        }
        case Cuda_Primitive_Type_Bezier:
        {
            // Bezier collision detection (simple approximation for demonstration)
            double3 V = normalize(*direction);
            double3 P = *origin;
            double t_min = 1e-6;
            double t_max = 1e10;

            // Iterate over the control points (simple approximation, you may need a more complex approach)
            for (int i = 0; i < primitive->data.bezier.degree; ++i) {
                double3 B = lerp(primitive->data.bezier.O1, primitive->data.bezier.O2, i / (double)primitive->data.bezier.degree);
                double r = primitive->data.bezier.R[i];
                double3 m = P - B;
                double b = dot(m, V);
                double c = dot(m, m) - r * r;
                if (c > 0.0 && b > 0.0) continue;

                double disc = b * b - c;
                if (disc < 0.0) continue;

                double t0 = -b - sqrt(disc);
                if (t0 < t_min || t0 > t_max) continue;

                collision->dist = t0;
                collision->C = P + V * t0;
                collision->N = normalize(collision->C - B);
                collision->front = (dot(V, collision->N) < 0);
                if (!collision->front) collision->N = -collision->N;
                collision->isCollide = true;
                collision->collide_primitive = primitive;
                break;
            }
            break;
        }
    }

    return collision->isCollide;
}
