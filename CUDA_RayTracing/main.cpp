#include "Renderer.h"

int main()
{
    Renderer renderer(1600, 900,
        1024, 576,
        "scene_mesh2.txt"
    );
    renderer.Render();
    return 0;
}