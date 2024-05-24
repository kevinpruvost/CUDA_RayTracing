#include "Renderer.h"

int main()
{
    Renderer renderer(1280, 720,
        1280, 720,
        "scene_mesh2.txt"
    );
    renderer.Render();
    return 0;
}