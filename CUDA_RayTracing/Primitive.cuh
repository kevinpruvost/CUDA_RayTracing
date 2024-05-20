#ifndef PRIMITIVE_CUH
#define PRIMITIVE_CUH

#include <cuda_runtime.h>

struct Cuda_Material
{
    float3 color, absor;
    double refl, refr;
    double diff, spec;
    double rindex;
    double drefl;
    uchar3 * texture;
    int texture_width, texture_height;
};

__device__ float3 operator+(const float3& a, const float3& b);
__device__ float3& operator+=(float3& a, const float3& b);
__device__ float3 operator*(const float3& a, float b);
__device__ float3 operator*(float b, const float3& a);

__device__ float3 GetMaterialSmoothPixel(Cuda_Material* material, float u, float v);

enum Cuda_Primitive_Type
{
    Cuda_Primitive_Type_Sphere,
    Cuda_Primitive_Type_Plane,
    Cuda_Primitive_Type_Square,
    Cuda_Primitive_Type_Cylinder,
    Cuda_Primitive_Type_Bezier,
    Cuda_Primitive_Type_Triangle
};

// Struct for Sphere
struct Cuda_Sphere {
    float3 O;  // Center of the sphere
    double R;   // Radius of the sphere
};

// Struct for Plane
struct Cuda_Plane {
    float3 N;  // Normal of the plane
    double R;   // Distance from origin
    float3 Dx, Dy;
};

// Struct for Square
struct Cuda_Square {
    float3 O;   // Origin of the square
    float3 Dx;  // Direction vector along one side
    float3 Dy;  // Direction vector along the other side
};

// Struct for Cylinder
struct Cuda_Cylinder {
    float3 O1;  // One end of the cylinder
    float3 O2;  // Other end of the cylinder
    double R;    // Radius of the cylinder
};

// Struct for Bezier
struct Cuda_Bezier {
    float3 O1;  // Start point
    float3 O2;  // End point
    float3 N;   // Normal
    float3 Nx;  // Tangent vector
    float3 Ny;  // Binormal vector
    int degree;  // Degree of the Bezier curve
    // Here we assume R and Z arrays have a fixed maximum size for simplicity
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
    // Add other primitives here (like triangle) if needed
};

struct Cuda_Primitive {
    bool isLightPrimitive;       // Indicates if the primitive is a light source
    Cuda_Material material;      // Material properties of the primitive
    Cuda_Primitive_Type type;    // Type of the primitive
    Cuda_Primitive_Data data;    // Data for the primitive
};

#define BIG_DIST 1e100

struct Cuda_Collision {
    bool isCollide;
    Cuda_Primitive* collide_primitive;
    float3 N, C;
    double dist;
    bool front;
};
__device__ Cuda_Collision InitCudaCollision();

__device__ bool intersect(Cuda_Primitive * primitive, float3 origin, float3 direction, Cuda_Collision* collision);

#endif