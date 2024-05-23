#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>
#include <vector>
#include "raytracer.h"
#include "Raytracer.cuh"
#include "Scene.cuh"
#include <memory>
#include "BmpSave.h"

#define MALLOC(ptr, size) assert(cudaMalloc(ptr, size) == cudaSuccess)
#define FREE(ptr) assert(cudaFree(ptr) == cudaSuccess)
#define MEMCPY(dst, src, size, kind) assert(cudaMemcpy(dst, src, size, kind) == cudaSuccess)

void FreeCudaScene(Cuda_Scene* cudaScene);
class SceneContainer
{
public:
    SceneContainer(Cuda_Scene* scene) : m_scene(scene) {}
    ~SceneContainer() { FreeCudaScene(m_scene); }

    Cuda_Scene* m_scene;
};

class SurfaceContainer
{
public:
    SurfaceContainer(cudaArray* textureArray) : m_surface(0) {
        // Create a surface object
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = textureArray;
        cudaCreateSurfaceObject(&m_surface, &resDesc);
    }
    ~SurfaceContainer() { cudaDestroySurfaceObject(m_surface); }

    cudaSurfaceObject_t m_surface;
};

class SettingsContainer
{
public:
    SettingsContainer(Settings * settings) : m_settings(settings) {
    }
    ~SettingsContainer() {
        FREE(m_settings);
    }
    Settings * m_settings;
};

class Renderer {
public:
    Renderer(int width, int height, int textureWidth, int textureHeight, const std::string & scene);
    ~Renderer();
    void Render();

private:
    void InitOpenGL();
    void InitCUDA();
    void CreateTexture();
    void RegisterCUDAResources();
    void UnregisterCUDAResources();
    void SetupImGui();
    void SetupQuad();
    void launchCudaKernel(cudaArray* textureArray, int texture_width, int texture_height, Raytracer * cpuScene);
    void ResetSceneInfos();
    void ResetSettings();
    void LoadScene(const std::string& scenePath);
    void GUI();
    void SaveTextureToBMP();

    cudaArray* textureArray;

    GLFWwindow* window;
    GLuint texture;
    cudaGraphicsResource* cudaTextureResource;
    int width, height;
    int texture_width, texture_height;

    GLuint vao, vbo, ebo;

    Raytracer raytracer;
    std::vector<float> frameTimes;
    std::string scenePath;
    int frameIndex;
    const int maxFrames;

    void ResetRendering();

    int segmentation;
    bool firstImage;
    bool generateOneImage;
    int x_progress;
    int y_progress;

    Settings settings;

    std::string outputName;

    std::unique_ptr<SceneContainer> m_sceneContainer;
    std::unique_ptr<SurfaceContainer> m_surfaceContainer;
    std::unique_ptr<SettingsContainer> m_settingsContainer;
};