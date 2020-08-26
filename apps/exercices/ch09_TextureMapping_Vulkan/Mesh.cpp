#include "Mesh.h"

void Mesh::setColor(SLCol4f color)
{
    for (size_t i = 0; i < C.size(); i++)
        C[i] = color;
}
/*
void Mesh::buildAABB(SLAABBox& aabb, const SLMat4f& wmNode)
{
    calcMinMax();
    aabb.fromOStoWS(minP, maxP, wmNode);
}
*/
void Mesh::calcMinMax()
{
    // init min & max points
    minP.set(FLT_MAX, FLT_MAX, FLT_MAX);
    maxP.set(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    // calc min and max point of all vertices
    for (SLulong i = 0; i < P.size(); ++i)
    {
        if (finalP((SLuint)i).x < minP.x) minP.x = finalP((SLuint)i).x;
        if (finalP((SLuint)i).x > maxP.x) maxP.x = finalP((SLuint)i).x;
        if (finalP((SLuint)i).y < minP.y) minP.y = finalP((SLuint)i).y;
        if (finalP((SLuint)i).y > maxP.y) maxP.y = finalP((SLuint)i).y;
        if (finalP((SLuint)i).z < minP.z) minP.z = finalP((SLuint)i).z;
        if (finalP((SLuint)i).z > maxP.z) maxP.z = finalP((SLuint)i).z;
    }
}
