#ifndef PRIMITIVE_CUH
#define PRIMITIVE_CUH

#include <cuda_runtime.h>
#include <helper_math.h>

struct Cuda_Material
{
    double3 color, absor;
    double refl, refr;
    double diff, spec;
    double rindex;
    double drefl;
    uchar3 * texture;
    int texture_width, texture_height;
};

__device__ double3 GetMaterialSmoothPixel(Cuda_Material* material, double u, double v);

enum Cuda_Primitive_Type
{
    Cuda_Primitive_Type_Sphere,
    Cuda_Primitive_Type_Plane,
    Cuda_Primitive_Type_Square,
    Cuda_Primitive_Type_Cylinder,
    Cuda_Primitive_Type_Bezier,
    Cuda_Primitive_Type_Triangle
};

struct Cuda_Triangle {
    double3 O1, O2, O3;
    double3 N;
};

// Struct for Sphere
struct Cuda_Sphere {
    double3 O;  // Center of the sphere
    double R;   // Radius of the sphere
    double3 De, Dc;
};

// Struct for Plane
struct Cuda_Plane {
    double3 N;  // Normal of the plane
    double R;   // Distance from origin
    double3 Dx, Dy;
};

// Struct for Square
struct Cuda_Square {
    double3 O;   // Origin of the square
    double3 Dx;  // Direction vector along one side
    double3 Dy;  // Direction vector along the other side
};

// Struct for Cylinder
struct Cuda_Cylinder {
    double3 O1;  // One end of the cylinder
    double3 O2;  // Other end of the cylinder
    double R;    // Radius of the cylinder
};

// Struct for Bezier
struct Cuda_Bezier {
    double3 O1;  // Start point
    double3 O2;  // End point
    double3 N;   // Normal
    double3 Nx;  // Tangent vector
    double3 Ny;  // Binormal vector
    double R_c;    // Radius of the cylinder based on the Bezier curve
    int degree;  // Degree of the Bezier curve
    // Here we assume R and Z arrays have a fixed maximum size for simplicity and only that have only R[degree] and Z[degree] data
    double R[10];  // Radii for control points (assuming a maximum degree)
    double Z[10];  // Z coordinates for control points
};

// Union of all possible primitive types
union Cuda_Primitive_Data {
    Cuda_Sphere sphere;
    Cuda_Plane plane;
    Cuda_Square square;
    Cuda_Cylinder cylinder;
    Cuda_Bezier bezier;
    Cuda_Triangle triangle;
    // Add other primitives here (like triangle) if needed
};

struct Cuda_Primitive {
    bool isLightPrimitive;       // Indicates if the primitive is a light source
    Cuda_Material material;      // Material properties of the primitive
    Cuda_Primitive_Type type;    // Type of the primitive
    Cuda_Primitive_Data data;    // Data for the primitive
};

struct Cuda_BVH
{
    Cuda_BVH* left;
    Cuda_BVH* right;
    double3 min;
    double3 max;
    Cuda_Primitive* primitive;
};

#define BIG_DIST 1e100

struct Cuda_Collision {
    bool isCollide;
    Cuda_Primitive* collide_primitive;
    double3 N, C;
    double dist;
    bool front;
};
__device__ Cuda_Collision InitCudaCollision();
__device__ double3 GetTextureColor(Cuda_Collision * collision);

__device__ bool intersect_primitives(Cuda_Primitive * primitive, const double3 * origin, const double3 * direction, Cuda_Collision* collision);
__device__ void BezierIntersect(Cuda_Primitive* primitive, const double3* origin, const double3* direction, Cuda_Collision* collision);
__device__ void CylinderIntersect(Cuda_Primitive* primitive, const double3* origin, const double3* direction, Cuda_Collision* collision);

#endif