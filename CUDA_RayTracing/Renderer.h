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

class Renderer {
public:
    Renderer(int width, int height, const std::string & scene);
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
    void launchCudaKernel(cudaArray* textureArray, int width, int height, Raytracer * cpuScene);
    void LoadScene(const std::string& scenePath);
    void GUI();

    GLFWwindow* window;
    GLuint texture;
    cudaGraphicsResource* cudaTextureResource;
    int width, height;

    GLuint vao, vbo, ebo;

    Raytracer raytracer;
    std::vector<float> frameTimes;
    std::string scenePath;
    int frameIndex;
    const int maxFrames;
    int resampling_size;

    void ResetRendering();

    int segmentation;
    bool firstImage;
    int x_progress;
    int y_progress;

    std::unique_ptr<SceneContainer> m_sceneContainer;
    std::unique_ptr<SurfaceContainer> m_surfaceContainer;
};