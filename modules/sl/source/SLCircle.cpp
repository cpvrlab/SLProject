#include <SLCircle.h>

SLCircle::SLCircle(SLAssetManager* assetMgr, SLstring name, SLMaterial* material)
  : SLPolyline(assetMgr, name)
{
    SLint   circlePoints = 60;
    SLfloat deltaPhi     = Utils::TWOPI / (SLfloat)circlePoints;

    SLVVec3f points;
    for (SLint i = 0; i < circlePoints; ++i)
    {
        SLVec2f c;
        c.fromPolar(1.0f, i * deltaPhi);
        points.push_back(SLVec3f(c.x, c.y, 0));
    }

    buildMesh(points, true, material);
}