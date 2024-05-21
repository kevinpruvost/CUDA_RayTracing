#include "Renderer.h"

int main()
{
    Renderer renderer(1600, 900,
        "scene1.txt"
    );
    renderer.Render();
    return 0;
}