#include "Primitive.cuh"

//__device__ bool IntersectInfiniteCylinder(const double3* origin, const double3* direction, Cuda_Bezier* bezier, double3* intersectionPoint1, double3 * intersectionPoint2)
//{
//    // Extract relevant parameters from the Cuda_Bezier struct
//    double3 O1 = bezier->O1;
//    double3 O2 = bezier->O2;
//    double3 N = bezier->N;
//    double R_c = bezier->R_c;
//
//    // Calculate the direction vector of the cylinder axis
//    double3 cylinderAxis = normalize(O2 - O1);
//
//    // Calculate the intersection of the ray with the infinite cylinder
//    double3 OC = *origin - O1;
//    double a = dot(*direction, *direction) - dot(*direction, cylinderAxis) * dot(*direction, cylinderAxis);
//    double b = 2.0 * (dot(*direction, OC) - dot(*direction, cylinderAxis) * dot(OC, cylinderAxis));
//    double c = dot(OC, OC) - dot(OC, cylinderAxis) * dot(OC, cylinderAxis) - R_c * R_c;
//
//    double discriminant = b * b - 4.0 * a * c;
//
//    if (discriminant < 0.0)
//        return false; // No intersection
//
//    double sqrtDiscriminant = sqrt(discriminant);
//    double t1 = (-b + sqrtDiscriminant) / (2.0 * a);
//    double t2 = (-b - sqrtDiscriminant) / (2.0 * a);
//
//    // Calculate intersection points
//    *intersectionPoint1 = *origin + t1 * (*direction);
//    *intersectionPoint2 = *origin + t2 * (*direction);
//
//    return true;
//}
//
//__device__ double2 DeCasteljau(double t, const double2* controlPoints, int degree) {
//    double2 points[12]; // Assuming degree + 1 <= 12
//    for (int i = 0; i <= degree - 1; i++) {
//        points[i] = controlPoints[i];
//    }
//    for (int r = 1; r <= degree - 1; r++) {
//        for (int i = 0; i <= degree - r - 1; i++) {
//            points[i] = (1.0f - t) * points[i] + t * points[i + 1];
//        }
//    }
//    return points[0];
//}
//
//__device__ bool IsPointOnCurve(const double2* controlPoints, int degree, double2 point, double tolerance) {
//    const int numGuesses = 100;
//    double t_values[numGuesses];
//
//    // Initialize t_values
//    for (int i = 0; i < numGuesses; ++i) {
//        t_values[i] = i / (double)(numGuesses - 1);
//    }
//
//    for (int i = 0; i < numGuesses; ++i) {
//        double t = t_values[i];
//        double2 bezierPoint = DeCasteljau(t, controlPoints, degree);
//        if (length(bezierPoint - point) < tolerance) {
//            return true;
//        }
//    }
//
//    return false;
//}
//
//__device__ bool CheckPointOnBezierCurve2D(const double3* P, const double3& R1, const double3& R2, double3* R, int degree, Cuda_Bezier* bezier)
//{
//    // Converts to a 2D problem
//    double3 normal = normalize(cross(R2 - R1, bezier->O1 - R1));
//    double3 axisX = normalize(R2 - R1);
//    double3 axisY = cross(normal, axisX);
//
//    double2 P2D = make_double2(dot(*P - R1, axisX), dot(*P - R1, axisY));
//
//    double2 R1_2D = make_double2(0, 0);
//    double2 R2_2D = make_double2(length(R2 - R1), 0);
//    double2 R_2D[10];
//    double2 controlPoints[12];
//    for (int i = 0; i < degree; i++) {
//        controlPoints[i + 1] = R_2D[i] = make_double2(dot(R[i] - R1, axisX), dot(R[i] - R1, axisY));
//    }
//
//    // Add the first and last control points
//    controlPoints[0] = R1_2D;
//    controlPoints[degree + 1] = R2_2D;
//
//    // Check if the point is on the Bezier curve
//    bool found = IsPointOnCurve(controlPoints, degree + 1, P2D, 1e-2);
//
//    return found;
//}
//
//__device__ void BezierIntersect(Cuda_Primitive* primitive, const double3* origin, const double3* direction, Cuda_Collision* collision)
//{
//    Cuda_Bezier* bezier = &primitive->data.bezier;
//    double3 V = normalize(*direction);
//
//    // First calculate if the ray intersects the infinite cylinder
//    double3 intersectionPointCylinder1;
//    double3 intersectionPointCylinder2;
//    if (IntersectInfiniteCylinder(origin, direction, bezier, &intersectionPointCylinder1, &intersectionPointCylinder2))
//    {
//        // Ray marching to test collision
//        int trials = 100;
//        double3 twoIntersectionsDir = intersectionPointCylinder2 - intersectionPointCylinder1;
//        double3 axis = normalize(bezier->O2 - bezier->O1);
//        double distance = length(bezier->O2 - bezier->O1);
//        for (int i = 0; i < trials; ++i)
//        {
//            double3 intersectionPointCylinder = intersectionPointCylinder1 + twoIntersectionsDir * ((double)i / (double)trials) ;
//            // Calculate the intersection point in the Bezier curve
//
//            // Calculate the first and last control point position on the axis with intersectionPointCylinder (O1 and O2 are the first and last control points)
//            // Calculate position of first control point based on distance from intersection point on axis
//            double distanceOnAxis = dot(bezier->O1 - intersectionPointCylinder, axis);
//            double3 R1 = intersectionPointCylinder + distanceOnAxis * axis;
//            double3 R2 = R1 + distance * axis;
//
//            double3 R[10];
//            double3 r1ToO1 = normalize(bezier->O1 - R1);
//            for (int j = 0; j < bezier->degree; j++)
//            {
//                R[j] = R1 + bezier->Z[j] * axis;
//                // Add perpendicular vector
//                R[j] += r1ToO1 * (bezier->R_c - bezier->R[j]);
//            }
//
//            // Calculate the intersection point in the Bezier curve
//            if (CheckPointOnBezierCurve2D(&intersectionPointCylinder, R1, R2, R, bezier->degree, bezier))
//            {
//                // Calculate the normal at the intersection point
//                double3 N = normalize(intersectionPointCylinder - bezier->O1);
//
//                // Calculate the distance from the origin to the intersection point
//                double dist = length(intersectionPointCylinder - *origin);
//
//                // Check if the intersection point is in front of the ray
//                bool front = dot(N, V) < 0.0;
//
//                // Update the collision struct
//                collision->isCollide = true;
//                collision->collide_primitive = primitive;
//                collision->C = intersectionPointCylinder;
//                collision->N = front ? N : -N;
//                collision->dist = dist;
//                collision->front = front;
//                return;
//            }
//        }
//    }
//}

__device__ void BezierIntersect(Cuda_Primitive* primitive, const double3* origin, const double3* direction, Cuda_Collision* collision)
{
    // Cut Bezier surface into multiple cylinders
    Cuda_Bezier* bezier = &primitive->data.bezier;

    double closest = 1e10;
    for (int i = 0; i < bezier->degree; ++i) {
        double3 point1;
        double3 point2;
        double R;
        double3 axis = normalize(bezier->O2 - bezier->O1);
        if (i == 0)
        {
            point1 = bezier->O1;
            point2 = bezier->O1 + bezier->Z[i] * axis;
            R = bezier->R_c;
        }
        else
        {
            point1 = bezier->O1 + bezier->Z[i - 1] * axis;
            point2 = bezier->O1 + bezier->Z[i] * axis;
            R = bezier->R[i];
        }

        Cuda_Cylinder cylinder;
        cylinder.O1 = point1;
        cylinder.O2 = point2;
        cylinder.R = R;
        Cuda_Primitive cylinderPrimitive;
        cylinderPrimitive.type = Cuda_Primitive_Type_Cylinder;
        cylinderPrimitive.data.cylinder = cylinder;
        Cuda_Collision tempCollision = InitCudaCollision();
        CylinderIntersect(&cylinderPrimitive, origin, direction, &tempCollision);
        if (tempCollision.isCollide && tempCollision.dist < closest)
        {
            closest = collision->dist;
            *collision = tempCollision;
            collision->collide_primitive = primitive;
        }
    }
}