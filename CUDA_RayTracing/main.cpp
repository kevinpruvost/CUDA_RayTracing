#include "Renderer.h"

int main()
{
    Renderer renderer(1600, 900,
        1280, 720,
        "scene2.txt"
    );
    renderer.RenderInParallel("frite_test.bmp", 16, 10, 50);
    return 0;
}