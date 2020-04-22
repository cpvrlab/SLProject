#ifndef SLCOORDAXISARROW_H
#define SLCOORDAXISARROW_H

#include <SLMesh.h>

class SLCoordAxisArrow : public SLMesh
{
public:
    SLCoordAxisArrow(SLAssetManager* assetMgr,
                     SLMaterial*     material        = nullptr,
                     SLfloat         arrowThickness  = 0.05f,
                     SLfloat         arrowHeadLenght = 0.2f,
                     SLfloat         arrowHeadWidth  = 0.1f);

    void buildMesh(SLMaterial* material = nullptr);

private:
    SLfloat _arrowThickness;  //!< Thickness of the arrow
    SLfloat _arrowHeadLength; //!< Lenght of the arrow head
    SLfloat _arrowHeadWidth;  //!< Width of the arrow head
};

#endif