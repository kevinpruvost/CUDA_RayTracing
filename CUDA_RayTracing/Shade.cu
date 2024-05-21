#include "Light.cuh"

__device__ double3 GetRandPointLight(double3 C, Cuda_Light* light)
{
    return light->O + light->Dx * (2 * frand() - 1.0) + light->Dy * (2 * frand() - 1.0);
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
            if (intersect(primitive, C, V, &tmp) && tmp.dist < dist)
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

            // if light ray collide any object, light source produce no shade to diffuse light
            for (int j = 0; j < primitivesCount; ++j)
            {
                Cuda_Primitive* primitive = &primitives[j];
                if (primitive == light->lightPrimitive) continue;
                Cuda_Collision tmp;
                if (intersect(primitive, C, V, &tmp) && tmp.dist < dist)
                {
                    shade += 1.0f;
                    break;
                }
            }
        }

        shade /= shade_quality;
        break;
    }
    }
    return shade;
}
