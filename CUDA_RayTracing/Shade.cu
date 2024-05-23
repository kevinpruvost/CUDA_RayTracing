#include "Light.cuh"

__device__ double3 GetRandPointLight(double3 C, Cuda_Light* light)
{
    return light->O + light->Dx * (2 * frand() - 1.0) + light->Dy * (2 * frand() - 1.0);
}

__device__ double3 GetRandPointLightSphere(double3 C, Cuda_Light* light)
{
    // Randomly sample spherical coordinates
    double z = 2.0 * frand() - 1.0;  // Random z in range [-1, 1]
    double theta = frand() * 2.0 * M_PI;  // Random theta in range [0, 2*pi]

    // Calculate the point on the unit sphere
    double r = sqrt(1.0 - z * z);
    double x = r * cos(theta);
    double y = r * sin(theta);

    // Scale by the light's radius and offset by the light's position
    return light->O + light->R * make_double3(x, y, z);
}

__device__ double CalnShade(double3 C, Cuda_Primitive * crashed_Primitive, Cuda_Light* light, Cuda_Primitive* primitives, int primitivesCount, int shade_quality)
{
    double shade = 0.0f;

    switch (light->type)
    {
    case Cuda_Light_Type_Point:
    {
        // light ray from diffuse point to light source
        double3 V = light->O - C;
        double dist = length(V);

        // if light ray collide any object, light source produce no shade to diffuse light
        for (int i = 0; i < primitivesCount; ++i)
        {
            if (&primitives[i] == crashed_Primitive) continue;
            Cuda_Primitive* primitive = &primitives[i];
            Cuda_Collision tmp = InitCudaCollision();
            intersect(primitive, &C, &V, &tmp);
            if (dist - tmp.dist > 1e-6)
            {
                return 0.0f;
            }
        }

        shade = 1.0f;
        break;
    }
    case Cuda_Light_Type_Square:
    {
        for (int i = 0; i < shade_quality; i++)
        {
            // sample a point light from light primitive
            double3 randO = GetRandPointLight(C, light);

            // light ray from diffuse point to point light
            double3 V = randO - C;
            double dist = length(V);

            int addShade = 1;
            // if light ray collide any object, light source produce no shade to diffuse light
            for (int j = 0; j < primitivesCount; ++j)
            {
                Cuda_Primitive* primitive = &primitives[j];
                if (primitive == crashed_Primitive || primitive == light->lightPrimitive) continue;
                Cuda_Collision tmp = InitCudaCollision();
                intersect(primitive, &C, &V, &tmp);
                if (dist - tmp.dist > 1e-6)
                {
                    addShade = 0;
                    break;
                }
            }
            shade += addShade;
        }
        shade /= shade_quality;
        break;
    }
    case Cuda_Light_Type_Sphere:
    {
        for (int i = 0; i < shade_quality; i++)
        {
            // sample a point light from light primitive
            double3 randO = GetRandPointLightSphere(C, light);

            // light ray from diffuse point to point light
            double3 V = randO - C;
            double dist = length(V);

            int addShade = 1;
            // if light ray collide any object, light source produce no shade to diffuse light
            for (int j = 0; j < primitivesCount; ++j)
            {
                Cuda_Primitive* primitive = &primitives[j];
                if (primitive == crashed_Primitive || primitive == light->lightPrimitive) continue;
                Cuda_Collision tmp = InitCudaCollision();
                intersect(primitive, &C, &V, &tmp);
                if (dist - tmp.dist > 1e-6)
                {
                    addShade = 0;
                    break;
                }
            }
            shade += addShade;
        }
        shade /= shade_quality;
        break;
    }
    }
    return shade;
}
