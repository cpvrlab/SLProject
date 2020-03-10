#ifndef SLCOORDAXISARROW_H
#define SLCOORDAXISARROW_H

#include <SLMesh.h>

class SLCoordAxisArrow : public SLMesh
{
public:
    SLCoordAxisArrow(SLVec4f arrowColor      = SLVec4f::GREEN,
                     SLfloat arrowThickness  = 0.05f,
                     SLfloat arrowHeadLenght = 0.2f,
                     SLfloat arrowHeadWidth  = 0.1f);

    void buildMesh();

private:
    SLfloat _arrowThickness;  //!< Thickness of the arrow
    SLfloat _arrowHeadLength; //!< Lenght of the arrow head
    SLfloat _arrowHeadWidth;  //!< Width of the arrow head
    SLVec4f _arrowColor;
};

#endif