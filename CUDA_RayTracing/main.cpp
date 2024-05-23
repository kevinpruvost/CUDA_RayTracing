#include "Renderer.h"

int main()
{
    Renderer renderer(800, 450,
        "scene_bezier.txt"
    );
    renderer.Render();
    return 0;
}