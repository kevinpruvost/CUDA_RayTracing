#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>

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

    void SetupQuad();


    GLFWwindow* window;
    GLuint texture;
    cudaGraphicsResource* cudaTextureResource;
    int width, height;

    GLuint vao, vbo, ebo;
};