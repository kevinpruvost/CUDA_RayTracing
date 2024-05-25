#include "Renderer.h"

int main()
{
    Renderer renderer(1600, 900,
//        3840, 2160,
        1920, 1080,
        "scene_final.txt"
    );
    renderer.settings.depthOfField.aperture = 70;
    renderer.settings.depthOfField.focalDistance = 168;
    renderer.Render();
    //renderer.RenderInParallel("scene_final.bmp", 32, 70, 168);
    //renderer.RenderInParallel("super_fine_one.bmp", 32, 1, 1.35);
    return 0;
}