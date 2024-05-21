#include "Scene.cuh"

__device__ const double SPEC_POWER = 20;
__device__ const int MAX_DREFL_DEP = 2;
__device__ const int MAX_RAYTRACING_DEP = 10;
__device__ const int HASH_FAC = 7;
__device__ const int HASH_MOD = 10000007;
__device__ const int NUM_RESAMPLE = 3;

__device__ double3 CalnDiffusion(Cuda_Scene* scene, Cuda_Collision * collide_primitive) {
    Cuda_Primitive* primitive = collide_primitive->collide_primitive;
    double3 color = primitive->material.color;

    // use texture if any
    if (primitive->material.texture != NULL) {
        color *= GetTextureColor(collide_primitive);
    }

    double3 ret = make_double3(0.0, 0.0, 0.0);

    for (int i = 0; i < scene->lightCount; ++i) {
        Cuda_Light* light = &scene->lights[i];
        double shade = CalnShade(collide_primitive->C, light, scene->primitives, scene->primitiveCount, int(16 * scene->camera.shade_quality));
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

//__device__ double3 CalnReflection(Cuda_Scene* scene, Cuda_Collision * collide_primitive, double3 ray_V, int dep) {
//    ray_V = reflect(ray_V, collide_primitive->N);
//    Cuda_Primitive* primitive = collide_primitive->collide_primitive;
//
//    // only reflection
//    if (primitive->material.drefl < 1e-6 || dep > MAX_DREFL_DEP) {
//        return traceRay(scene, collide_primitive->C, ray_V, dep + 1) * primitive->material.color * primitive->material.refl;
//    }
//    // diffuse reflection (fuzzy reflection)
//    else {
//        // Unit circle perpendicular to ray_V
//        double3 baseX = GetAnVerticalVector(ray_V);
//        double3 baseY = normalize(cross(ray_V, baseX));
//
//        // scale the circle according to drefl (fuzzy) value
//        baseX *= primitive->material.drefl;
//        baseY *= primitive->material.drefl;
//
//        // ADD BLUR
//        double2 xy = primitive->material.blur->GetXY();
//        double3 fuzzy_V = ray_V + baseX * xy.x + baseY * xy.y;
//
//        return traceRay(scene, collide_primitive->C, fuzzy_V, dep + 1) * primitive->material.color * primitive->material.refl;
//    }
//}
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

__device__ Cuda_Collision intersect(Cuda_Scene* scene, double3 origin, double3 direction)
{
    Cuda_Collision collision = InitCudaCollision();
    Cuda_Collision empty = collision;

    for (int i = 0; i < scene->primitiveCount; i++)
    {
        Cuda_Primitive* primitive = &scene->primitives[i];
        Cuda_Collision temp_collision = empty;
        if (intersect(primitive, origin, direction, &temp_collision))
        {
            if (temp_collision.dist < collision.dist)
            {
                collision = temp_collision;
            }
        }
    }
    return collision;
}

__device__ const int max_depth = 5;
__device__ double3 traceRay(Cuda_Scene* scene, double u, double v, int x, int y, int depth)
{
    double3 color = make_double3(0.0f, 0.0f, 0.0f);

    Cuda_Camera* c = &scene->camera;
    double3 origin = c->O;
    // N + Dy * (2 * i / H - 1) + Dx * (2 * j / W - 1)
    double3 direction = c->N + c->Dy * (2 * (double)y / c->H - 1) + c->Dx * (2 * (double)x / c->W - 1);

    Cuda_Collision collision;
    collision = intersect(scene, origin, direction);
    if (collision.isCollide)
    {
        Cuda_Primitive * prim = collision.collide_primitive;
        if (prim->isLightPrimitive)
            color = collision.collide_primitive->material.color;
        else {
            if (prim->material.diff > 1e-6 || prim->material.spec > 1e-6) {
                color += CalnDiffusion(scene, &collision);
            }
            if (prim->material.refl > 1e-6) {
                //color += CalnReflection(prim, direction, depth);
            }
            if (prim->material.refr > 1e-6) {
                //color += CalnRefraction(prim, direction, depth);
            }
        }
    }
    else
    {
        color = (v < 0.5) ? scene->backgroundColor_top : scene->backgroundColor_bottom;
    }
    color.x = fmin(1.0, color.x);
    color.y = fmin(1.0, color.y);
    color.z = fmin(1.0, color.z);
    return color;
}