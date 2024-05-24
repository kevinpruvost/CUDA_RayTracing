#include "Renderer.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>
#include <filesystem>
#include "raytracer.h"

Renderer::Renderer(int width, int height, int tWidth, int tHeight, const std::string& scene)
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
    , generateOneImage{ true }
    , texture_width{ tWidth }
    , texture_height{ tHeight }
{
    LoadScene(scenePath);
    frameTimes.resize(maxFrames, 0.0f);
    InitOpenGL();
    InitCUDA();
    CreateTexture();
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
    ResetRendering();
    ResetSceneInfos();
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

    // Block resize
    glfwSetWindowSizeLimits(window, width, height, width, height);

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
    if (texture != 0) {
        glDeleteTextures(1, &texture);
        UnregisterCUDAResources();
    }

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texture_width, texture_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
    RegisterCUDAResources();
    SetupQuad();
    m_surfaceContainer.reset();
    ResetRendering();
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
        cudaTextureResource = nullptr;
    }
}

void Renderer::ModifyViewport(int w, int h)
{
    width = w;
    height = h;
    glViewport(0, 0, width, height);
    glfwSetWindowSize(window, w, h);
    glfwSetWindowSizeLimits(window, width, height, width, height);
}

void Renderer::SaveTextureToBMP()
{
    std::vector<unsigned char> pixels(texture_width * texture_height * 3);
    glBindTexture(GL_TEXTURE_2D, texture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    // Reverse pixels
    std::vector<unsigned char> pixelsReversed(texture_width * texture_height * 3);
    for (int i = 0; i < pixels.size(); ++i) {
        pixelsReversed[i] = pixels[pixels.size() - i - 1];
    }
    BmpSave::SaveBMP(outputName, pixels.data(), texture_width, texture_height);
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

    if (vao != 0) glDeleteVertexArrays(1, &vao);
    if (ebo != 0) glDeleteBuffers(1, &ebo);
    if (vbo != 0) glDeleteBuffers(1, &vbo);

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
    if (generateOneImage)
    {
        int percentage = (double)y_progress / texture_height * segmentation * segmentation + (double)x_progress / texture_width * segmentation;
        if (firstImage == false) percentage = 100;
        ImGui::Text("Image Generation: %d%%", percentage);
    }

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
    if (ImGui::Button("Reload"))
    {
        LoadScene(scenesStr[currentSceneIdx]);
    }

    // Define the available resampling sizes
    static const char* resamplingOptions[] = { "1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024" };
    static int currentResamplingIdx = 0; // Index of the current resampling size

    ImGui::Separator();
    ImGui::Text("Window/Texture Settings");
    ImGui::BeginChild("TextureFrame", ImVec2(0, 100), true, ImGuiWindowFlags_NoResize);
    {
        // Define the available resampling sizes
        static const char* textureSizes[] = { "1280x720", "1600x900", "1920x1080", "2560x1440", "3840x2160" };
        int textureSizeIdx = 0; // Index of the current resampling size

        switch (texture_width)
        {
        case 1280:
            textureSizeIdx = 0;
            break;
        case 1600:
            textureSizeIdx = 1;
            break;
        case 1920:
            textureSizeIdx = 2;
            break;
        case 2560:
            textureSizeIdx = 3;
            break;
        case 3840:
            textureSizeIdx = 4;
            break;
        }
        // Display combo box to select texture size
        if (ImGui::Combo("Texture Size", &textureSizeIdx, textureSizes, IM_ARRAYSIZE(textureSizes))) {
            // Update texture size based on the selected option
            switch (textureSizeIdx)
            {
            case 0:
                texture_width = 1280;
                texture_height = 720;
                break;
            case 1:
                texture_width = 1600;
                texture_height = 900;
                break;
            case 2:
                texture_width = 1920;
                texture_height = 1080;
                break;
            case 3:
                texture_width = 2560;
                texture_height = 1440;
                break;
            case 4:
                texture_width = 3840;
                texture_height = 2160;
                break;
            }
            CreateTexture();
        }

        int viewportSizeIdx = 0;

        switch (width)
        {
        case 1280:
            viewportSizeIdx = 0;
            break;
        case 1600:
            viewportSizeIdx = 1;
            break;
        case 1920:
            viewportSizeIdx = 2;
            break;
        case 2560:
            viewportSizeIdx = 3;
            break;
        case 3840:
            viewportSizeIdx = 4;
            break;
        }

        // Display combo box to select viewport size
        if (ImGui::Combo("Viewport Size", &viewportSizeIdx, textureSizes, IM_ARRAYSIZE(textureSizes))) {
            // Update viewport size based on the selected option
            switch (viewportSizeIdx)
            {
            case 0:
                ModifyViewport(1280, 720);
                break;
            case 1:
                ModifyViewport(1600, 900);
                break;
            case 2:
                ModifyViewport(1920, 1080);
                break;
            case 3:
                ModifyViewport(2560, 1440);
                break;
            case 4:
                ModifyViewport(3840, 2160);
                break;
            }
        }
    }
    ImGui::EndChild();

    // Create a subframe for settings
    ImGui::Separator();
    ImGui::Text("Post Processing Settings");
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
                settings.depthOfField.focalDistance = focalDistance;
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
}

void Renderer::Render()
{
    double lastTime = glfwGetTime();
    long long int frameCount = 0;
    GLuint shaderProgram = createShaderProgram(shaderVert, shaderFrag);

    // Get uniform location for aspect ratio
    GLuint aspectRatioLoc = glGetUniformLocation(shaderProgram, "aspectRatio");

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
        launchCudaKernel(textureArray, texture_width, texture_height, &raytracer);

        cudaGraphicsUnmapResources(1, &cudaTextureResource, 0);

        // Compute aspect ratios
        float windowAspect = static_cast<float>(width) / height;
        float textureAspect = static_cast<float>(texture_width) / texture_height;

        // Render the texture to the screen
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);

        // Set the aspect ratio uniform
        glUniform2f(aspectRatioLoc, windowAspect, textureAspect);

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

void Renderer::RenderInParallel(const std::string& output, int resampling, double aperture, double focalDistance)
{
    // Renders image then saves it to bmp and close program
    // Map CUDA resource
    cudaGraphicsMapResources(1, &cudaTextureResource, 0);
    cudaGraphicsSubResourceGetMappedArray(&textureArray, cudaTextureResource, 0, 0);

    // Launch CUDA kernel to write to the texture (pseudo-code, replace with actual kernel call)
    settings.resampling_size = resampling;
    settings.depthOfField.enabled = true;
    settings.depthOfField.aperture = aperture;
    settings.depthOfField.focalDistance = focalDistance;
    outputName = output;
    launchCudaKernelParallel(textureArray, texture_width, texture_height, &raytracer);

    cudaGraphicsUnmapResources(1, &cudaTextureResource, 0);
    SaveTextureToBMP();
}
