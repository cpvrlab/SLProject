#include <SLCoordAxisArrow.h>

SLCoordAxisArrow::SLCoordAxisArrow(SLVec4f arrowColor,
                                   SLfloat arrowThickness,
                                   SLfloat arrowHeadLenght,
                                   SLfloat arrowHeadWidth) : SLMesh("Coord-Axis-Arrow Mesh")
{
    _arrowColor      = arrowColor;
    _arrowThickness  = arrowThickness;
    _arrowHeadLength = arrowHeadLenght;
    _arrowHeadWidth  = arrowHeadWidth;
    buildMesh();
}

void SLCoordAxisArrow::buildMesh()
{
    // allocate new vectors of SLMesh
    P.clear();
    N.clear();
    C.clear();
    Tc.clear();
    I16.clear();

    //Set one default material index
    //In SLMesh::init this will be set automatically to SLMaterial::diffuseAttrib
    mat(nullptr);

    SLushort i = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0;
    SLfloat  t = _arrowThickness * 0.5f;
    SLfloat  h = _arrowHeadLength;
    SLfloat  w = _arrowHeadWidth * 0.5f;

    // clang-format off
    P.push_back(SLVec3f( t,  t, t)); C.push_back(_arrowColor); v1=i; i++;
    P.push_back(SLVec3f( t,1-h, t)); C.push_back(_arrowColor); v2=i; i++;
    P.push_back(SLVec3f( t,1-h,-t)); C.push_back(_arrowColor); v3=i; i++;
    P.push_back(SLVec3f( t,  t,-t)); C.push_back(_arrowColor); v4=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2); I16.push_back(v1); I16.push_back(v4); I16.push_back(v3);

    P.push_back(SLVec3f(-t,  t, t)); C.push_back(_arrowColor); v1=i; i++;
    P.push_back(SLVec3f(-t,1-h, t)); C.push_back(_arrowColor); v2=i; i++;
    P.push_back(SLVec3f( t,1-h, t)); C.push_back(_arrowColor); v3=i; i++;
    P.push_back(SLVec3f( t,  t, t)); C.push_back(_arrowColor); v4=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2); I16.push_back(v1); I16.push_back(v4); I16.push_back(v3);

    P.push_back(SLVec3f(-t,  t,-t)); C.push_back(_arrowColor); v1=i; i++;
    P.push_back(SLVec3f(-t,1-h,-t)); C.push_back(_arrowColor); v2=i; i++;
    P.push_back(SLVec3f(-t,1-h, t)); C.push_back(_arrowColor); v3=i; i++;
    P.push_back(SLVec3f(-t,  t, t)); C.push_back(_arrowColor); v4=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2); I16.push_back(v1); I16.push_back(v4); I16.push_back(v3);

    P.push_back(SLVec3f( t,  t,-t)); C.push_back(_arrowColor); v1=i; i++;
    P.push_back(SLVec3f( t,1-h,-t)); C.push_back(_arrowColor); v2=i; i++;
    P.push_back(SLVec3f(-t,1-h,-t)); C.push_back(_arrowColor); v3=i; i++;
    P.push_back(SLVec3f(-t,  t,-t)); C.push_back(_arrowColor); v4=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2); I16.push_back(v1); I16.push_back(v4); I16.push_back(v3);

    P.push_back(SLVec3f( w,1-h, w)); C.push_back(_arrowColor); v1=i; i++;
    P.push_back(SLVec3f( 0,  1, 0)); C.push_back(_arrowColor); v2=i; i++;
    P.push_back(SLVec3f( w,1-h,-w)); C.push_back(_arrowColor); v3=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2);

    P.push_back(SLVec3f( w,1-h, w)); C.push_back(_arrowColor); v1=i; i++;
    P.push_back(SLVec3f(-w,1-h, w)); C.push_back(_arrowColor); v2=i; i++;
    P.push_back(SLVec3f( 0,  1, 0)); C.push_back(_arrowColor); v3=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2);
    P.push_back(SLVec3f(-w,1-h, w)); C.push_back(_arrowColor); v1=i; i++;
    P.push_back(SLVec3f(-w,1-h,-w)); C.push_back(_arrowColor); v2=i; i++;
    P.push_back(SLVec3f( 0,  1, 0)); C.push_back(_arrowColor); v3=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2);

    P.push_back(SLVec3f(-w,1-h,-w)); C.push_back(_arrowColor); v1=i; i++;
    P.push_back(SLVec3f( w,1-h,-w)); C.push_back(_arrowColor); v2=i; i++;
    P.push_back(SLVec3f( 0,  1, 0)); C.push_back(_arrowColor); v3=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2);

    P.push_back(SLVec3f( w,1-h, w)); C.push_back(_arrowColor); v1=i; i++;
    P.push_back(SLVec3f( w,1-h,-w)); C.push_back(_arrowColor); v2=i; i++;
    P.push_back(SLVec3f(-w,1-h,-w)); C.push_back(_arrowColor); v3=i; i++;
    P.push_back(SLVec3f(-w,1-h, w)); C.push_back(_arrowColor); v4=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2); I16.push_back(v1); I16.push_back(v4); I16.push_back(v3);

    P.push_back(SLVec3f( t, -t, t)); C.push_back(_arrowColor); v1=i; i++;
    P.push_back(SLVec3f( t, -t,-t)); C.push_back(_arrowColor); v2=i; i++;
    P.push_back(SLVec3f(-t, -t,-t)); C.push_back(_arrowColor); v3=i; i++;
    P.push_back(SLVec3f(-t, -t, t)); C.push_back(_arrowColor); v4=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2); I16.push_back(v1); I16.push_back(v4); I16.push_back(v3);
}
