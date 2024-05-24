#include "Renderer.h"

int main()
{
    Renderer renderer(1600, 900,
        3840, 2160,
        "scene_mesh2.txt"
    );
    renderer.Render();
    return 0;
}