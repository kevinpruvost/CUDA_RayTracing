#include "Renderer.h"

#ifndef _DEBUG
#undef assert
#define assert(x) if (!(x)) { printf("Assertion failed: %s\n", #x); exit(1); }
#endif

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

        Cuda_Primitive* defaultLightPrim = nullptr;
        MEMCPY(&cudaLights[i].lightPrimitive, &defaultLightPrim, sizeof(Cuda_Primitive*), cudaMemcpyHostToDevice);

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
            MEMCPY(&cudaPrimitives[i].data.bezier.N, &bezier->N, sizeof(Vector3), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.bezier.Nx, &bezier->Nx, sizeof(Vector3), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.bezier.Ny, &bezier->Ny, sizeof(Vector3), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.bezier.R_c, &bezier->R_c, sizeof(double), cudaMemcpyHostToDevice);
            double* rData = bezier->R.data();
            double* zData = bezier->Z.data();
            MEMCPY(&cudaPrimitives[i].data.bezier.R, rData, (bezier->degree + 1) * sizeof(double), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.bezier.Z, zData, (bezier->degree + 1) * sizeof(double), cudaMemcpyHostToDevice);

            type = Cuda_Primitive_Type_Bezier;
        }
        else if (dynamic_cast<Triangle*>(currentPrimitive) != nullptr)
        {
            type = Cuda_Primitive_Type_Triangle;

            Triangle * triangle = dynamic_cast<Triangle*>(currentPrimitive);
            MEMCPY(&cudaPrimitives[i].data.triangle.O1, &triangle->O1, sizeof(Vector3), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.triangle.O2, &triangle->O2, sizeof(Vector3), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.triangle.O3, &triangle->O3, sizeof(Vector3), cudaMemcpyHostToDevice);
            MEMCPY(&cudaPrimitives[i].data.triangle.N, &triangle->N, sizeof(Vector3), cudaMemcpyHostToDevice);
        }
        MEMCPY(&cudaPrimitives[i].type, &type, sizeof(Cuda_Primitive_Type), cudaMemcpyHostToDevice);

        currentPrimitive = currentPrimitive->GetNext();
        ++i;
    }

    return cudaPrimitives;
}

#include <unordered_map>

void LinkPrimitivesToLight(Cuda_Scene* scene, Primitive* primHead, Light* lightHead) {
    int primCount, lightCount;
    MEMCPY(&primCount, &scene->primitiveCount, sizeof(int), cudaMemcpyDeviceToHost);
    MEMCPY(&lightCount, &scene->lightCount, sizeof(int), cudaMemcpyDeviceToHost);

    Cuda_Primitive* cudaPrimitives;
    Cuda_Light* cudaLights;
    MEMCPY(&cudaPrimitives, &scene->primitives, sizeof(Cuda_Primitive*), cudaMemcpyDeviceToHost);
    MEMCPY(&cudaLights, &scene->lights, sizeof(Cuda_Light*), cudaMemcpyDeviceToHost);

    // Create a map from CPU primitive pointers to GPU primitive pointers
    std::unordered_map<Primitive*, Cuda_Primitive*> primMap;
    Primitive* currentPrim = primHead;
    for (int i = 0; i < primCount; ++i) {
        primMap[currentPrim] = &cudaPrimitives[i];
        currentPrim = currentPrim->GetNext();
    }

    // Link light primitives
    Light* currentLight = lightHead;
    for (int j = 0; j < lightCount; ++j) {
        if (currentLight->lightPrimitive) {
            Primitive* lightPrimitive = currentLight->lightPrimitive;
            if (primMap.find(lightPrimitive) != primMap.end()) {
                Cuda_Primitive* cudaLightPrimitive = primMap[lightPrimitive];
                MEMCPY(&cudaLights[j].lightPrimitive, &cudaLightPrimitive, sizeof(Cuda_Primitive*), cudaMemcpyHostToDevice);
            }
        }
        currentLight = currentLight->GetNext();
    }
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

    // Link Primitives to Lights
    LinkPrimitivesToLight(scene, sceneCpu->scene.primitive_head, sceneCpu->light_head);

    // Random seeds
    unsigned long randomSeeds[BLOCK_SIZE * BLOCK_SIZE];
    for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) {
        randomSeeds[i] = rand();
    }
    unsigned long* d_seeds;
    MALLOC(&d_seeds, BLOCK_SIZE * BLOCK_SIZE * sizeof(unsigned long));
    MEMCPY(d_seeds, randomSeeds, BLOCK_SIZE * BLOCK_SIZE * sizeof(unsigned long), cudaMemcpyHostToDevice);
    MEMCPY(&(scene->seeds), &d_seeds, sizeof(unsigned long*), cudaMemcpyHostToDevice);
    return scene;
}

void FreeCudaScene(Cuda_Scene* cudaScene)
{
    Cuda_Scene scene_freer;
    MEMCPY(&scene_freer, cudaScene, sizeof(Cuda_Scene), cudaMemcpyDeviceToHost);

    // Free random seeds
    FREE(scene_freer.seeds);

    // Free primitives
    FREE(scene_freer.primitives);

    // Free Lights
    FREE(scene_freer.lights);

    // Free scene
    FREE(cudaScene);
}

// Renderer function to launch the kernel and work with surfaces
void Renderer::launchCudaKernel(cudaArray* textureArray, int texture_width, int texture_height, Raytracer * cpuScene)
{
    if (m_settingsContainer.get() == nullptr)
    {
        ResetSettings();
    }

    if (m_surfaceContainer.get() == nullptr)
    {
        m_surfaceContainer.reset(new SurfaceContainer(textureArray));
    }

    if (m_sceneContainer.get() == nullptr)
    {
        ResetSceneInfos();
    }

    // Launch the kernel via the wrapper function
    if (firstImage && settings.resampling_size > 8)
    {
        launchRayTraceKernel(m_surfaceContainer->m_surface, texture_width, texture_height, width, height, m_sceneContainer->m_scene, x_progress, y_progress, texture_width / segmentation, texture_height / segmentation, m_settingsContainer->m_settings);
        x_progress += texture_width / segmentation;
        if (x_progress >= texture_width)
        {
            x_progress = 0;
            y_progress += texture_height / segmentation;
        }
        if (y_progress >= texture_height)
        {
            firstImage = false;
            x_progress = 0;
            y_progress = 0;
        }
    }
    else if (!generateOneImage || firstImage)
    {
        launchRayTraceKernel(m_surfaceContainer->m_surface, texture_width, texture_height, width, height, m_sceneContainer->m_scene, 0, 0, texture_width, texture_height, m_settingsContainer->m_settings);
        firstImage = false;
    }
}

void Renderer::ResetSceneInfos()
{
    m_surfaceContainer.reset(new SurfaceContainer(textureArray));

    m_sceneContainer.reset(new SceneContainer(createCudaSceneFromCPUScene(&raytracer, width, height)));
    cudaDeviceSetLimit(cudaLimitStackSize, 4096 * 16);
}

void Renderer::ResetSettings()
{
    ResetRendering();
    Settings * dsettings = nullptr;
    MALLOC(&dsettings, sizeof(Settings));
    MEMCPY(dsettings, &settings, sizeof(Settings), cudaMemcpyHostToDevice);
    m_settingsContainer.reset(new SettingsContainer(dsettings));
    m_surfaceContainer.reset(new SurfaceContainer(textureArray));
}