#include "Renderer.h"

int main()
{
    Renderer renderer(1600, 900,
        "scene_bezier.txt"
    );
    renderer.Render();
    return 0;
}