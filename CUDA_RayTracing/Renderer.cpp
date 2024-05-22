#include "Renderer.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>
#include <filesystem>
#include "raytracer.h"

Renderer::Renderer(int width, int height, const std::string& scene)
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
    , scenePath{scene}
    , firstImage{true}
    , segmentation{ 10 }
    , x_progress{ 0 }
    , y_progress{ 0 }
    , m_sceneContainer(nullptr)
    , m_surfaceContainer(nullptr)
    , outputName{ "output.bmp" }
    , generateOneImage{false}
{
    LoadScene(scenePath);
    frameTimes.resize(maxFrames, 0.0f);
    InitOpenGL();
    InitCUDA();
    CreateTexture();
    RegisterCUDAResources();
    SetupQuad();
    SetupImGui();

    // Settings
    settings.resampling_size = 1;
    settings.depthOfField.enabled = false;
    settings.depthOfField.aperture = 10.0f;
    settings.depthOfField.focalDistance = 10.0f;
}

void Renderer::LoadScene(const std::string& scene)
{
    raytracer = Raytracer();
    raytracer.SetInput(scene);
    raytracer.CreateAll();
    scenePath = scene;
    m_sceneContainer.reset(nullptr);
    ResetRendering();
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

void Renderer::SaveTextureToBMP()
{
    std::vector<unsigned char> pixels(width * height * 3);
    glBindTexture(GL_TEXTURE_2D, texture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    // Reverse pixels
    std::vector<unsigned char> pixelsReversed(width * height * 3);
    for (int i = 0; i < pixels.size(); ++i) {
        pixelsReversed[i] = pixels[pixels.size() - i - 1];
    }
    BmpSave::SaveBMP(outputName, pixels.data(), width, height);
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

void Renderer::GUI()
{
    // Display FPS
    ImGui::SetNextWindowSize(ImVec2(420, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(5, 5), ImGuiCond_FirstUseEver);
    ImGui::Begin("Details");
    ImGui::Text("Framerate: %.1f", ImGui::GetIO().Framerate);
    ImGui::Text("1 Frame each %.5f seconds", ImGui::GetIO().DeltaTime);
    ImGui::PlotLines("FPS", frameTimes.data(), maxFrames, frameIndex, nullptr, 0.0f, 100.0f, ImVec2(0, 80));

    // Display combo box to load other scenes
    std::vector<std::string> scenesStr;
    // Load strings from resources path
    int currentSceneIdx = 0;
    std::string folderPath = "./"; // Replace with your folder path
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (entry.path().extension() == ".txt") {
            std::string scene = entry.path().filename().string();
            scenesStr.push_back(scene);
            if (scene == scenePath) {
                currentSceneIdx = scenesStr.size() - 1;
            }
        }
    }

    ImGui::Separator();
    ImGui::Text("Save Image");
    if (ImGui::Button("Save")) {
        SaveTextureToBMP();
    }
    // Set output name
    ImGui::InputText("Output Name", outputName.data(), outputName.size());
    ImGui::Checkbox("Generate One Image", &generateOneImage);

    ImGui::Separator();
    ImGui::Text("Select Scene:");
    if (ImGui::BeginCombo("##Select Scene:", scenesStr[currentSceneIdx].c_str())) {
        for (int i = 0; i < scenesStr.size(); i++) {
            bool is_selected = (currentSceneIdx == i);
            if (ImGui::Selectable(scenesStr[i].c_str(), is_selected)) {
                currentSceneIdx = i;
                LoadScene(scenesStr[currentSceneIdx]);
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    // Define the available resampling sizes
    static const char* resamplingOptions[] = { "1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024" };
    static int currentResamplingIdx = 0; // Index of the current resampling size

    // Create a subframe for settings
    ImGui::Separator();
    ImGui::Text("Settings");
    ImGui::BeginChild("SettingsFrame", ImVec2(0, 100), true, ImGuiWindowFlags_NoResize);
    bool settingsChanged = false;
    {
        // Display combo box to select resampling size
        if (ImGui::Combo("Resampling Size", &currentResamplingIdx, resamplingOptions, IM_ARRAYSIZE(resamplingOptions))) {
            // Update resampling_size based on the selected option
            settings.resampling_size = std::stoi(resamplingOptions[currentResamplingIdx]);
            settingsChanged = true;
        }

        // Display checkbox to enable/disable depth of field
        if (ImGui::Checkbox("Depth of Field", &settings.depthOfField.enabled)) {
            settingsChanged = true;
        }
        if (settings.depthOfField.enabled) {
            // Display slider to adjust aperture
            float aperture = settings.depthOfField.aperture;
            if (ImGui::DragFloat("Aperture", &aperture, 0.1f, 0.5f, 1000.0f)) {
                settings.depthOfField.aperture = aperture;
                settingsChanged = true;
            }

            // Display slider to adjust focal distance
            float focalDistance = settings.depthOfField.focalDistance;
            if (ImGui::DragFloat("Focal Distance", &focalDistance, 0.1f, 0.5f, 1000.0f)) {
                settingsChanged = true;
                focalDistance = settings.depthOfField.focalDistance;
            }
        }
    }
    if (settingsChanged) ResetSettings();
    ImGui::EndChild(); // End Settings

    ImGui::End();
}

void Renderer::ResetRendering()
{
    firstImage = true;
    x_progress = 0;
    y_progress = 0;
    m_surfaceContainer.reset(nullptr);
}

void Renderer::Render()
{
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

        GUI();

        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteProgram(shaderProgram);
}