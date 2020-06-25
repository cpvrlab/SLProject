#include "Mesh.h"

void Mesh::setColor(SLCol4f color)
{
    for (size_t i = 0; i < C.size(); i++)
        C[i] = color;
}
