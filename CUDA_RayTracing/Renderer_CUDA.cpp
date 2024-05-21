#include "Renderer.h"
#include "Raytracer.cuh"
#include "Scene.cuh"
#include "raytracer.h"
#include <memory>

#ifndef _DEBUG
#undef assert
#define assert(x) if (!(x)) { printf("Assertion failed: %s\n", #x); exit(1); }
#endif

#define MALLOC(ptr, size) assert(cudaMalloc(ptr, size) == cudaSuccess)
#define FREE(ptr) assert(cudaFree(ptr) == cudaSuccess)
#define MEMCPY(dst, src, size, kind) assert(cudaMemcpy(dst, src, size, kind) == cudaSuccess)

void createCudaCameraFromCPUCamera(Cuda_Camera * cudaCamera, Camera* camera, int width, int height)
{
    // Camera position
    MEMCPY(&cudaCamera->O, &camera->O, sizeof(Vector3), cudaMemcpyHostToDevice);
    MEMCPY(&cudaCamera->N, &camera->N, sizeof(Vector3), cudaMemcpyHostToDevice);
    MEMCPY(&cudaCamera->Dx, &camera->Dx, sizeof(Vector3), cudaMemcpyHostToDevice);
    MEMCPY(&cudaCamera->Dy, &camera->Dy, sizeof(Vector3), cudaMemcpyHostToDevice);

    // Pixels
    MEMCPY(&cudaCamera->W, &width, sizeof(int), cudaMemcpyHostToDevice);
    MEMCPY(&cudaCamera->H, &height, sizeof(int), cudaMemcpyHostToDevice);

    MEMCPY(&cudaCamera->shade_quality, &camera->shade_quality, sizeof(double), cudaMemcpyHostToDevice);
    MEMCPY(&cudaCamera->drefl_quality, &camera->drefl_quality, sizeof(double), cudaMemcpyHostToDevice);
    MEMCPY(&cudaCamera->max_photons, &camera->max_photons, sizeof(int), cudaMemcpyHostToDevice);
    MEMCPY(&cudaCamera->emit_photons, &camera->emit_photons, sizeof(int), cudaMemcpyHostToDevice);
    MEMCPY(&cudaCamera->sample_photons, &camera->sample_photons, sizeof(int), cudaMemcpyHostToDevice);
    MEMCPY(&cudaCamera->sample_dist, &camera->sample_dist, sizeof(double), cudaMemcpyHostToDevice);
}

Cuda_Light * createCudaLightsFromCPULights(Light* lights, int * lightCount)
{
    *lightCount = 0;
    for (Light* currentLight = lights; currentLight != nullptr; currentLight = currentLight->GetNext())
        ++(*lightCount);

    Cuda_Light* cudaLights = nullptr;
    MALLOC(&cudaLights, (*lightCount) * sizeof(Cuda_Light));

    Light* currentLight = lights;
    int i = 0;
    while (currentLight != nullptr)
    {
        // Light properties
        MEMCPY(&cudaLights[i].sample, &currentLight->sample, sizeof(int), cudaMemcpyHostToDevice);
        MEMCPY(&cudaLights[i].color, &currentLight->color, sizeof(Color), cudaMemcpyHostToDevice);
        MEMCPY(&cudaLights[i].O, &currentLight->O, sizeof(Vector3), cudaMemcpyHostToDevice);
        Cuda_Light_Type type;
        if (dynamic_cast<PointLight*>(currentLight) != nullptr)
        {
            type = Cuda_Light_Type_Point;
        }
        else if (dynamic_cast<SquareLight*>(currentLight) != nullptr)
        {
            type = Cuda_Light_Type_Square;
            MEMCPY(&cudaLights[i].Dx, &(dynamic_cast<SquareLight*>(currentLight))->Dx, sizeof(Vector3), cudaMemcpyHostToDevice);
            MEMCPY(&cudaLights[i].Dy, &(dynamic_cast<SquareLight*>(currentLight))->Dy, sizeof(Vector3), cudaMemcpyHostToDevice);
        }
        else if (dynamic_cast<SphereLight*>(currentLight) != nullptr)
        {
            type = Cuda_Light_Type_Sphere;
            MEMCPY(&cudaLights[i].R, &(dynamic_cast<SphereLight*>(currentLight))->R, sizeof(double), cudaMemcpyHostToDevice);
        }
        MEMCPY(&cudaLights[i].type, &type, sizeof(Cuda_Light_Type), cudaMemcpyHostToDevice);

        //cudaLights->lightPrimitive = nullptr;

        currentLight = currentLight->GetNext();
        ++i;
    }
    return cudaLights;
}

Cuda_Primitive* createCudaPrimitivesFromCPUPrimitives(Primitive* primitives, int * primCount)
{
    *primCount = 0;
    for (Primitive* currentPrimitive = primitives; currentPrimitive != nullptr; currentPrimitive = currentPrimitive->GetNext())
        ++(*primCount);

    Cuda_Primitive* cudaPrimitives = nullptr;
    MALLOC(&cudaPrimitives, (*primCount) * sizeof(Cuda_Primitive));

    Primitive * currentPrimitive = primitives;
    int i = 0;
    while (currentPrimitive != nullptr) {
        // Primitive properties
        bool isLightPrimitive = currentPrimitive->IsLightPrimitive();
        MEMCPY(&cudaPrimitives[i].isLightPrimitive, &isLightPrimitive, sizeof(bool), cudaMemcpyHostToDevice);

        // Material
        {
            //struct Cuda_Material
            //{
            //    float3 color, absor;
            //    double refl, refr;
            //    double diff, spec;
            //    double rindex;
            //    double drefl;
            //    uchar3 * texture;
            //    int texture_width, texture_height;
            //};
            Material* material = currentPrimitive->GetMaterial();
            MEMCPY(&cudaPrimitives[i].material.color, &material->color, sizeof(Color), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].material.absor, &material->absor, sizeof(Color), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].material.refl, &material->refl, sizeof(double), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].material.refr, &material->refr, sizeof(double), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].material.diff, &material->diff, sizeof(double), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].material.spec, &material->spec, sizeof(double), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].material.rindex, &material->rindex, sizeof(double), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].material.drefl, &material->drefl, sizeof(double), cudaMemcpyHostToDevice);
            if (material->texture != nullptr) {
                int texture_width = material->texture->GetW();
                int texture_height = material->texture->GetH();
                MEMCPY(&cudaPrimitives[i].material.texture_height, &texture_height, sizeof(int), cudaMemcpyHostToDevice);
                MEMCPY(&cudaPrimitives[i].material.texture_width, &texture_width, sizeof(int), cudaMemcpyHostToDevice);

                uchar3* texture = new uchar3[texture_width * texture_height];
                for (int i = 0; i < texture_width; ++i)
                    for (int j = 0; j < texture_height; ++j)
                        memcpy(&texture[i * (int)texture_height + j], &material->texture->ima[j][i], sizeof(uchar3));
                uchar3* texturePtr;
                MALLOC(&texturePtr, texture_width * texture_height * sizeof(uchar3));
                MEMCPY(texturePtr, texture, texture_width * texture_height * sizeof(uchar3), cudaMemcpyHostToDevice);
                MEMCPY(&(cudaPrimitives[i].material.texture), &texturePtr, sizeof(uchar3*), cudaMemcpyHostToDevice);
                delete texture;
            }
            else {
                cudaMemset(&cudaPrimitives[i].material.texture, 0, sizeof(uchar3**));
                cudaMemset(&cudaPrimitives[i].material.texture_width, 0, sizeof(double));
                cudaMemset(&cudaPrimitives[i].material.texture_height, 0, sizeof(double));
            }
        }

        // Primitive type
        Cuda_Primitive_Type type;
        if (dynamic_cast<Sphere*>(currentPrimitive) != nullptr)
        {
            Sphere * sphere = dynamic_cast<Sphere*>(currentPrimitive);
            MEMCPY(&cudaPrimitives[i].data.sphere.O, &sphere->O, sizeof(Vector3), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.sphere.R, &sphere->R, sizeof(double), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.sphere.De, &sphere->De, sizeof(Vector3), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.sphere.Dc, &sphere->Dc, sizeof(Vector3), cudaMemcpyHostToDevice);
            type = Cuda_Primitive_Type_Sphere;
        }
        else if (dynamic_cast<Plane*>(currentPrimitive) != nullptr)
        {
            Plane * plane = dynamic_cast<Plane*>(currentPrimitive);
            MEMCPY(&cudaPrimitives[i].data.plane.N, &plane->N, sizeof(Vector3), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.plane.R, &plane->R, sizeof(double), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.plane.Dx, &plane->Dx, sizeof(Vector3), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.plane.Dy, &plane->Dy, sizeof(Vector3), cudaMemcpyHostToDevice);
            type = Cuda_Primitive_Type_Plane;
        }
        else if (dynamic_cast<Square*>(currentPrimitive) != nullptr)
        {
            Square * square = dynamic_cast<Square*>(currentPrimitive);
            MEMCPY(&cudaPrimitives[i].data.square.O, &square->O, sizeof(Vector3), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.square.Dx, &square->Dx, sizeof(Vector3), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.square.Dy, &square->Dy, sizeof(Vector3), cudaMemcpyHostToDevice);
            type = Cuda_Primitive_Type_Square;
        }
        else if (dynamic_cast<Cylinder*>(currentPrimitive) != nullptr)
        {
            Cylinder * cylinder = dynamic_cast<Cylinder*>(currentPrimitive);
            MEMCPY(&cudaPrimitives[i].data.cylinder.O1, &cylinder->O1, sizeof(Vector3), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.cylinder.O2, &cylinder->O2, sizeof(Vector3), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.cylinder.R, &cylinder->R, sizeof(double), cudaMemcpyHostToDevice);
            type = Cuda_Primitive_Type_Cylinder;
        }
        else if (dynamic_cast<Bezier*>(currentPrimitive) != nullptr)
        {
            Bezier * bezier = dynamic_cast<Bezier*>(currentPrimitive);
            MEMCPY(&cudaPrimitives[i].data.bezier.O1, &bezier->O1, sizeof(Vector3), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.bezier.O2, &bezier->O2, sizeof(Vector3), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.bezier.degree, &bezier->degree, sizeof(int), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.bezier.R, bezier->R.data(), bezier->R.size() * sizeof(double), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.bezier.Z, bezier->Z.data(), bezier->Z.size() * sizeof(double), cudaMemcpyHostToDevice);

            type = Cuda_Primitive_Type_Bezier;
        }
        //else if (dynamic_cast<Triangle*>(currentPrimitive) != nullptr)
        //{
        //    type = Cuda_Primitive_Type_Triangle;
        //}
        MEMCPY(&cudaPrimitives[i].type, &type, sizeof(Cuda_Primitive_Type), cudaMemcpyHostToDevice);

        currentPrimitive = currentPrimitive->GetNext();
        ++i;
    }

    return cudaPrimitives;
}

Cuda_Scene* createCudaSceneFromCPUScene(Raytracer * sceneCpu, int width, int height)
{
    Cuda_Scene * scene = nullptr;
    MALLOC(&scene, sizeof(Cuda_Scene));

    // Background color
    MEMCPY(&scene->backgroundColor_bottom, &sceneCpu->background_color_bottom, sizeof(Color), cudaMemcpyHostToDevice);
    MEMCPY(&scene->backgroundColor_top, &sceneCpu->background_color_top, sizeof(Color), cudaMemcpyHostToDevice);

    // Camera
    createCudaCameraFromCPUCamera(&scene->camera, sceneCpu->camera, width, height);

    // Lights
    int lightCount;
    Cuda_Light * lights = createCudaLightsFromCPULights(sceneCpu->light_head, &lightCount);
    MEMCPY(&(scene->lights), &lights, sizeof(Cuda_Light*), cudaMemcpyHostToDevice);
    MEMCPY(&(scene->lightCount), &lightCount, sizeof(int), cudaMemcpyHostToDevice);

    // Primitives
    int primitiveCount;
    Cuda_Primitive * primitives = createCudaPrimitivesFromCPUPrimitives(sceneCpu->scene.primitive_head, &primitiveCount);
    MEMCPY(&(scene->primitives), &primitives, sizeof(Cuda_Primitive*), cudaMemcpyHostToDevice);
    MEMCPY(&(scene->primitiveCount), &primitiveCount, sizeof(int), cudaMemcpyHostToDevice);

    return scene;
}

void FreeCudaScene(Cuda_Scene* cudaScene)
{
    Cuda_Scene scene_freer;
    MEMCPY(&scene_freer, cudaScene, sizeof(Cuda_Scene), cudaMemcpyDeviceToHost);

    // Free primitives
    FREE(scene_freer.primitives);

    // Free Lights
    FREE(scene_freer.lights);

    // Free scene
    FREE(cudaScene);
}

class SceneContainer
{
public:
    SceneContainer(Cuda_Scene* scene) : m_scene(scene) {}
    ~SceneContainer() { FreeCudaScene(m_scene); }

    Cuda_Scene* m_scene;
};

// Renderer function to launch the kernel and work with surfaces
void Renderer::launchCudaKernel(cudaArray* textureArray, int width, int height, Raytracer * cpuScene)
{
    static std::unique_ptr<SceneContainer> sceneContainer(nullptr);

    // Create a surface object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = textureArray;

    cudaSurfaceObject_t surfaceObject = 0;
    cudaCreateSurfaceObject(&surfaceObject, &resDesc);

    if (sceneContainer.get() == nullptr)
        sceneContainer.reset(new SceneContainer(createCudaSceneFromCPUScene(cpuScene, width, height)));

    // Launch the kernel via the wrapper function
    launchRayTraceKernel(surfaceObject, width, height, sceneContainer->m_scene);

    // Clean up
    cudaDestroySurfaceObject(surfaceObject);
}