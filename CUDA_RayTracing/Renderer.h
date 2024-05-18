#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>
#include <vector>

class Renderer {
public:
    Renderer(int width, int height);
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
    void launchCudaKernel(cudaArray* textureArray, int width, int height);

    GLFWwindow* window;
    GLuint texture;
    cudaGraphicsResource* cudaTextureResource;
    int width, height;

    GLuint vao, vbo, ebo;

    std::vector<float> frameTimes;
    int frameIndex;
    const int maxFrames;
};