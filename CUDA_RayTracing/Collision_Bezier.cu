#include "Primitive.cuh"

__device__ bool IntersectInfiniteCylinder(const double3* origin, const double3* direction, Cuda_Bezier* bezier, double3* intersectionPoint1, double3 * intersectionPoint2)
{
    // Extract relevant parameters from the Cuda_Bezier struct
    double3 O1 = bezier->O1;
    double3 O2 = bezier->O2;
    double3 N = bezier->N;
    double R_c = bezier->R_c;

    // Calculate the direction vector of the cylinder axis
    double3 cylinderAxis = normalize(O2 - O1);

    // Calculate the intersection of the ray with the infinite cylinder
    double3 OC = *origin - O1;
    double a = dot(*direction, *direction) - dot(*direction, cylinderAxis) * dot(*direction, cylinderAxis);
    double b = 2.0 * (dot(*direction, OC) - dot(*direction, cylinderAxis) * dot(OC, cylinderAxis));
    double c = dot(OC, OC) - dot(OC, cylinderAxis) * dot(OC, cylinderAxis) - R_c * R_c;

    double discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0)
        return false; // No intersection

    double t1 = (-b + sqrt(discriminant)) / (2.0 * a);
    double t2 = (-b - sqrt(discriminant)) / (2.0 * a);

    // Calculate intersection points
    double3 P1 = *origin + t1 * (*direction);
    double3 P2 = *origin + t2 * (*direction);

    // Check if the intersection point lies outside the cylinder segment
    double3 OP1 = P1 - O1;
    double3 OP2 = P2 - O1;

    double proj1 = dot(OP1, cylinderAxis);
    double proj2 = dot(OP2, cylinderAxis);

    bool intersect1 = (proj1 >= 0.0 && proj1 <= length(O2 - O1));
    bool intersect2 = (proj2 >= 0.0 && proj2 <= length(O2 - O1));

    if (intersect1 && intersect2) {
        *intersectionPoint1 = P1;
        *intersectionPoint2 = P2;
        return true; // Both intersections are valid
    }
    else if (intersect1) {
        *intersectionPoint1 = P1;
        return true; // Only the first intersection is valid
    }
    else if (intersect2) {
        *intersectionPoint2 = P2;
        return true; // Only the second intersection is valid
    }

    return true;
}

__device__ double2 DeCasteljau(double t, const double2* controlPoints, int degree) {
    double2 points[12]; // Assuming degree + 1 <= 12
    for (int i = 0; i <= degree - 1; i++) {
        points[i] = controlPoints[i];
    }
    for (int r = 1; r <= degree - 1; r++) {
        for (int i = 0; i <= degree - r - 1; i++) {
            points[i] = (1.0f - t) * points[i] + t * points[i + 1];
        }
    }
    return points[0];
}

__device__ double DistanceToRay(double t, const double2* controlPoints, int degree, double2 rayStart, double2 rayDir) {
    double2 bezierPoint = DeCasteljau(t, controlPoints, degree);
    double t_ray = dot(bezierPoint - rayStart, rayDir) / dot(rayDir, rayDir);
    double2 rayPoint = rayStart + t_ray * rayDir;
    return length(bezierPoint - rayPoint);
}

__device__ double RefineIntersection(double t_guess, const double2* controlPoints, int degree, double2 rayStart, double2 rayDir, double tolerance) {
    double t = t_guess;
    for (int iter = 0; iter < 10; ++iter) {
        double d = DistanceToRay(t, controlPoints, degree, rayStart, rayDir);
        if (d < tolerance) {
            return t;
        }
        double grad = (DistanceToRay(t + tolerance, controlPoints, degree, rayStart, rayDir) - d) / tolerance;
        if (fabs(grad) < 1e-10) {  // Avoid division by zero
            break;
        }
        t -= d / grad;
        t = fmax(0.0, fmin(1.0, t));  // Clamp t to [0, 1]
    }
    return t;
}

__device__ void FindIntersections2D(const double2* controlPoints, int degree, double2 rayStart, double2 rayDir, double2* intersection, int* foundIntersection) {
    const int numGuesses = 10;
    const double tolerance = 1e-5;
    double t_values[numGuesses];
    double closest_distance = 1e10;

    // Initialize t_values
    for (int i = 0; i < numGuesses; ++i) {
        t_values[i] = i / (double)(numGuesses - 1);
    }

    for (int i = 0; i < numGuesses; ++i) {
        double t_guess = t_values[i];
        double t_intersection = RefineIntersection(t_guess, controlPoints, degree, rayStart, rayDir, tolerance);

        if (0.0 <= t_intersection && t_intersection <= 1.0) {
            double d = DistanceToRay(t_intersection, controlPoints, degree, rayStart, rayDir);
            if (d < tolerance && d < closest_distance) {
                *intersection = DeCasteljau(t_intersection, controlPoints, degree);
                *foundIntersection = 1;
                closest_distance = d;
            }
        }
    }
}

__device__ bool IntersectBezierCurve2D(const double3 * origin, const double3 * direction, const double3 * P, const double3& R1, const double3& R2, double3* R, int degree, double3* intersectionPoint, Cuda_Bezier* bezier)
{
    // Converts to a 2D problem
    double3 normal = normalize(cross(R2 - R1, bezier->O1 - R1));
    double3 axisX = normalize(R2 - R1);
    double3 axisY = cross(normal, axisX);

    double2 origin2D = make_double2(dot(*origin - R1, axisX), dot(*origin - R1, axisY));
    double2 direction2D = normalize(make_double2(dot(*direction, axisX), dot(*direction, axisY)));
    double2 P2D = make_double2(dot(*P - R1, axisX), dot(*P - R1, axisY));

    double2 R1_2D = make_double2(0, 0);
    double2 R2_2D = make_double2(length(R2 - R1), 0);
    double2 R_2D[10];
    double2 controlPoints[12];
    for (int i = 0; i < degree; i++)
    {
        controlPoints[i+1] = R_2D[i] = make_double2(dot(R[i] - R1, axisX), dot(R[i] - R1, axisY));
    }

    // Calculate the intersection point in 2D
    controlPoints[0] = R1_2D;
    controlPoints[degree+1] = R2_2D;
    double2 intersections;
    int found = 0;
    FindIntersections2D(controlPoints, degree+2, P2D, direction2D, &intersections, &found);

    // Return the first valid intersection
    if (found == 1)
    {
        double3 intersection3D = R1 + intersections.x * axisX + intersections.y * axisY;
        *intersectionPoint = intersection3D;
        return true;
    }
    return false;
}


__device__ void BezierIntersect(Cuda_Primitive* primitive, const double3* origin, const double3* direction, Cuda_Collision* collision)
{
    Cuda_Bezier* bezier = &primitive->data.bezier;
    double3 V = normalize(*direction);

    // First calculate if the ray intersects the infinite cylinder
    double3 intersectionPointCylinder1;
    double3 intersectionPointCylinder2;
    if (IntersectInfiniteCylinder(origin, direction, bezier, &intersectionPointCylinder1, &intersectionPointCylinder2))
    {
        double3 intersectionPointCylinder = intersectionPointCylinder1;
        // Calculate the intersection point in the Bezier curve
        double3 axis = normalize(bezier->O2 - bezier->O1);

        // Calculate the first and last control point position on the axis with intersectionPointCylinder (O1 and O2 are the first and last control points)
        double distance = length(bezier->O2 - bezier->O1);
        // Calculate position of first control point based on distance from intersection point on axis
        double distanceOnAxis = dot(bezier->O1 - intersectionPointCylinder, axis);
        double3 R1 = intersectionPointCylinder + distanceOnAxis * axis;
        double3 R2 = R1 + distance * axis;

        double3 R[10];
        double3 r1ToO1 = normalize(bezier->O1 - R1);
        for (int i = 0; i < bezier->degree; i++)
        {
            R[i] = R1 + bezier->Z[i] * axis;
            // Add perpendicular vector
            R[i] += r1ToO1 * (bezier->R_c - bezier->R[i]);
        }

        // Calculate the intersection point in the Bezier curve
        double3 intersectionPoint;
        if (IntersectBezierCurve2D(origin, direction, &intersectionPointCylinder, R1, R2, R, bezier->degree, &intersectionPoint, bezier))
        {
            // Calculate the normal at the intersection point
            double3 N = normalize(intersectionPoint - bezier->O1);

            // Calculate the distance from the origin to the intersection point
            double dist = length(intersectionPoint - *origin);

            // Check if the intersection point is in front of the ray
            bool front = dot(N, V) < 0.0;

            // Update the collision struct
            collision->isCollide = true;
            collision->collide_primitive = primitive;
            collision->C = intersectionPoint;
            collision->N = front ? N : -N;
            collision->dist = dist;
            collision->front = front;
        }
    }
}

//__device__ void BezierIntersect(Cuda_Primitive* primitive, const double3* origin, const double3* direction, Cuda_Collision* collision)
//{
//    // Cut Bezier surface into multiple cylinders
//    Cuda_Bezier* bezier = &primitive->data.bezier;
//
//    double closest = 1e10;
//    for (int i = 0; i < bezier->degree; ++i) {
//        double3 point1;
//        double3 point2;
//        double R;
//        double3 axis = normalize(bezier->O2 - bezier->O1);
//        if (i == 0)
//        {
//            point1 = bezier->O1;
//            point2 = bezier->O1 + bezier->Z[i] * axis;
//            R = bezier->R_c;
//        }
//        else
//        {
//            point1 = bezier->O1 + bezier->Z[i - 1] * axis;
//            point2 = bezier->O1 + bezier->Z[i] * axis;
//            R = bezier->R[i];
//        }
//
//        Cuda_Cylinder cylinder;
//        cylinder.O1 = point1;
//        cylinder.O2 = point2;
//        cylinder.R = R;
//        Cuda_Primitive cylinderPrimitive;
//        cylinderPrimitive.type = Cuda_Primitive_Type_Cylinder;
//        cylinderPrimitive.data.cylinder = cylinder;
//        Cuda_Collision tempCollision = InitCudaCollision();
//        CylinderIntersect(&cylinderPrimitive, origin, direction, &tempCollision);
//        if (tempCollision.isCollide && tempCollision.dist < closest)
//        {
//            closest = collision->dist;
//            *collision = tempCollision;
//            collision->collide_primitive = primitive;
//        }
//    }
//}