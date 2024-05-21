#include "Scene.cuh"

#define SPEC_POWER 20
#define MAX_DREFL_DEP 2
#define HASH_MOD 10000007
#define HASH_FAC 7

// Blur function
//std::pair<double, double> ExpBlur::GetXY()
//{
//    double x, y;
//    x = ran();
//    // x in [0, 1), but with a higher prob to be a small value
//    x = pow(2, x) - 1;
//    y = rand();
//    return std::pair<double, double>(x * cos(y), x * sin(y));
//}

__device__ double2 GetBlur()
{
    double x, y;
    x = frand();
    // x in [0, 1), but with a higher prob to be a small value
    // x = pow(2, x) - 1;
    y = frand();
    return make_double2(x * cosf(y), x * sinf(y));
}

__device__ double3 CalnDiffusion(Cuda_Scene* scene, Cuda_Collision* collide_primitive) {
    Cuda_Primitive* primitive = collide_primitive->collide_primitive;
    double3 color = primitive->material.color;

    // use texture if any
    if (primitive->material.texture != NULL) {
        color *= GetTextureColor(collide_primitive);
    }

    double3 ret = make_double3(0.0, 0.0, 0.0);

    for (int i = 0; i < scene->lightCount; ++i) {
        Cuda_Light* light = &scene->lights[i];
        double shade = CalnShade(collide_primitive->C, primitive, light, scene->primitives, scene->primitiveCount, int(16 * scene->camera.shade_quality));
        if (shade < 1e-6) {
            continue;
        }

        // now the light produces some shade on this diffuse point

        // R: light ray from diffuse point to the light
        double3 R = normalize(light->O - collide_primitive->C);
        double dotted = dot(R, collide_primitive->N);
        if (dotted > 1e-6) {
            // diffuse light
            if (primitive->material.diff > 1e-6) {
                double diff = primitive->material.diff * dotted * shade;
                ret += color * light->color * diff;
            }
            // specular light
            if (primitive->material.spec > 1e-6) {
                double spec = primitive->material.spec * pow(dotted, SPEC_POWER) * shade;
                ret += color * light->color * spec;
            }
        }
    }
    return ret;
}

__device__ double3 CalnReflection(Cuda_Scene* scene, Cuda_Collision* collide_primitive, double3 ray_V, int dep)
{
    ray_V = reflect(ray_V, collide_primitive->N);
    Cuda_Primitive* primitive = collide_primitive->collide_primitive;

    // only reflection
    if (primitive->material.drefl < 1e-6 || dep > MAX_DREFL_DEP) {
        return traceRay(scene, collide_primitive->C, ray_V, dep + 1) * primitive->material.color * primitive->material.refl;
    }
    // diffuse reflection (fuzzy reflection)
    else {
        // TODO: NEED TO IMPLEMENT

        // Unit *circle* perpendicular to ray_V.
        // This is different from sampling from a unit sphere -- when projecting the sphere
        // to this circle the points are not uniformly distributed.
        // However, considering the ExpBlur, this approximation may be justified.

        double3 baseX = GetAnVerticalVector(ray_V);
        double3 baseY = normalize(cross(ray_V, baseX));

        // scale the circle according to drefl (fuzzy) value
        baseX *= primitive->material.drefl;
        baseY *= primitive->material.drefl;

        // ADD BLUR
        double2 xy = GetBlur();
        double3 fuzzy_V = ray_V + baseX * xy.x + baseY * xy.y;

        return traceRay(scene, collide_primitive->C, fuzzy_V, dep + 1) * primitive->material.color * primitive->material.refl;
    }
}
//
//__device__ double3 CalnRefraction(Cuda_Scene* scene, Cuda_Collision * collide_primitive, double3 ray_V, int dep) {
//    Cuda_Primitive* primitive = collide_primitive->collide_primitive;
//    double n = primitive->material.rindex;
//    if (collide_primitive->front) {
//        n = 1.0 / n;
//    }
//
//    ray_V = refract(ray_V, collide_primitive->N, n);
//    double3 rcol = traceRay(scene, collide_primitive->C, ray_V, dep + 1);
//
//    if (collide_primitive->front) {
//        return rcol * primitive->material.refr;
//    }
//
//    double3 absor = primitive->material.absor * -collide_primitive->dist;
//    double3 trans = make_double3(exp(absor.x), exp(absor.y), exp(absor.z));
//    return rcol * trans * primitive->material.refr;
//}