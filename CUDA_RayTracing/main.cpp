#include "Renderer.h"

int main()
{
    Renderer renderer(1600, 900,
//        3840, 2160,
        1280, 720,
        "scene_final.txt"
    );
    renderer.Render();
    //renderer.RenderInParallel("super_fine_one.bmp", 32, 1, 1.35);
    return 0;
}