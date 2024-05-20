#include "Renderer.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>
#include "raytracer.h"

Renderer::Renderer(int width, int height)
    : width(width)
    , height(height)
    , window(nullptr)
    , texture(0)
    , cudaTextureResource(nullptr)
    , vao(0)
    , vbo(0)
    , ebo(0)
    , frameIndex(0)
    , maxFrames(100)
{
    frameTimes.resize(maxFrames, 0.0f);
    InitOpenGL();
    InitCUDA();
    CreateTexture();
    RegisterCUDAResources();
    SetupQuad();
    SetupImGui();
}

Renderer::~Renderer()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    UnregisterCUDAResources();
    if (vao != 0) glDeleteVertexArrays(1, &vao);
    if (ebo != 0) glDeleteBuffers(1, &ebo);
    if (vbo != 0) glDeleteBuffers(1, &vbo);
    glfwDestroyWindow(window);
    glfwTerminate();
}

void Renderer::InitOpenGL()
{
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, "CUDA OpenGL Renderer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        exit(EXIT_FAILURE);
    }

    glfwSwapInterval(1);
    glViewport(0, 0, width, height);
}

void Renderer::InitCUDA()
{
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Renderer::CreateTexture()
{
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::RegisterCUDAResources()
{
    cudaError_t err = cudaGraphicsGLRegisterImage(&cudaTextureResource, texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "Failed to register CUDA graphics resource: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Renderer::UnregisterCUDAResources()
{
    if (cudaTextureResource) {
        cudaGraphicsUnregisterResource(cudaTextureResource);
    }
}

void Renderer::SetupQuad()
{
    float quadVertices[] = {
        // Positions   // TexCoords
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 1.0f
    };

    GLuint indices[] = {
        0, 1, 2,
        2, 3, 0
    };

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Renderer::SetupImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");
}

static constexpr const char* shaderVert = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

static constexpr const char* shaderFrag = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screenTexture;

void main() {
    FragColor = texture(screenTexture, TexCoord);
}
)";

static GLuint createShaderProgram(const char* vert, const char* frag)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vert, nullptr);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &frag, nullptr);
    glCompileShader(fragmentShader);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

void Renderer::Render()
{
    Raytracer raytracer;
    raytracer.SetInput("./scene1.txt");
    raytracer.CreateAll();

    double lastTime = glfwGetTime();
    long long int frameCount = 0;
    GLuint shaderProgram = createShaderProgram(shaderVert, shaderFrag);

    while (!glfwWindowShouldClose(window))
    {
        // Calculate FPS
        double currentTime = glfwGetTime();
        double deltaTime = currentTime - lastTime;

        if (deltaTime >= 1.0f / 60.0f) { // If one second has passed, update FPS counter
            float fps = frameCount / static_cast<float>(deltaTime);

            // Update frameTimes and reset counters
            frameTimes[frameIndex] = fps;
            frameIndex = (frameIndex + 1) % maxFrames;
            frameCount = 0;
            lastTime = currentTime;
        }

        frameCount++;

        // Start the ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Map CUDA resource
        cudaArray* textureArray;
        cudaGraphicsMapResources(1, &cudaTextureResource, 0);
        cudaGraphicsSubResourceGetMappedArray(&textureArray, cudaTextureResource, 0, 0);

        // Launch CUDA kernel to write to the texture (pseudo-code, replace with actual kernel call)
        launchCudaKernel(textureArray, width, height, &raytracer);

        cudaGraphicsUnmapResources(1, &cudaTextureResource, 0);

        // Render the texture to the screen
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);

        glBindTexture(GL_TEXTURE_2D, texture);

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        glBindTexture(GL_TEXTURE_2D, 0);

        // Display FPS
        ImGui::Begin("Performance");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::PlotLines("FPS", frameTimes.data(), maxFrames, frameIndex, nullptr, 0.0f, 100.0f, ImVec2(0, 80));
        ImGui::End();

        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteProgram(shaderProgram);
}