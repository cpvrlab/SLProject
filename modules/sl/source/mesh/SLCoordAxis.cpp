//#############################################################################
//  File:      SLCoordAxis.cpp
//  Date:      June 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLCoordAxis.h>

//-----------------------------------------------------------------------------
//! SLAxis::SLAxis ctor with the arrow dimensions
SLCoordAxis::SLCoordAxis(SLAssetManager* assetMgr,
                         SLfloat         arrowThickness,
                         SLfloat         arrowHeadLenght,
                         SLfloat         arrowHeadWidth) : SLMesh(assetMgr, "Coord-Axis Mesh")
{
    _arrowThickness  = arrowThickness;
    _arrowHeadLength = arrowHeadLenght;
    _arrowHeadWidth  = arrowHeadWidth;
    buildMesh();
}
//-----------------------------------------------------------------------------
//! SLCoordAxis::buildMesh fills in the underlying arrays from the SLMesh object
void SLCoordAxis::buildMesh()
{
    // allocate new vectors of SLMesh
    P.clear();
    N.clear();
    C.clear();
    UV[0].clear();
    UV[1].clear();
    I16.clear();
    I32.clear();

    // Set one default material index
    // In SLMesh::init this will be set automatically to SLMaterial::diffuseAttrib
    mat(nullptr);

    SLushort i = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0;
    SLfloat  t = _arrowThickness * 0.5f;
    SLfloat  h = _arrowHeadLength;
    SLfloat  w = _arrowHeadWidth * 0.5f;

    // predefined normals
    SLCol4f r = SLCol4f::RED;
    SLCol4f g = SLCol4f::GREEN;
    SLCol4f b = SLCol4f::BLUE;

    // clang-format off
    // arrow towards +x
    P.push_back(SLVec3f(  t, t, t)); C.push_back(r); v1=i; i++;
    P.push_back(SLVec3f(1-h, t, t)); C.push_back(r); v2=i; i++;
    P.push_back(SLVec3f(1-h, t,-t)); C.push_back(r); v3=i; i++;
    P.push_back(SLVec3f(  t, t,-t)); C.push_back(r); v4=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v4);

    P.push_back(SLVec3f(  t,-t, t)); C.push_back(r); v1=i; i++;
    P.push_back(SLVec3f(1-h,-t, t)); C.push_back(r); v2=i; i++;
    P.push_back(SLVec3f(1-h, t, t)); C.push_back(r); v3=i; i++;
    P.push_back(SLVec3f(  t, t, t)); C.push_back(r); v4=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v4);

    P.push_back(SLVec3f(  t,-t,-t)); C.push_back(r); v1=i; i++;
    P.push_back(SLVec3f(1-h,-t,-t)); C.push_back(r); v2=i; i++;
    P.push_back(SLVec3f(1-h,-t, t)); C.push_back(r); v3=i; i++;
    P.push_back(SLVec3f(  t,-t, t)); C.push_back(r); v4=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v4);

    P.push_back(SLVec3f(  t, t,-t)); C.push_back(r); v1=i; i++;
    P.push_back(SLVec3f(1-h, t,-t)); C.push_back(r); v2=i; i++;
    P.push_back(SLVec3f(1-h,-t,-t)); C.push_back(r); v3=i; i++;
    P.push_back(SLVec3f(  t,-t,-t)); C.push_back(r); v4=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v4);

    P.push_back(SLVec3f(1-h, w, w)); C.push_back(r); v1=i; i++;
    P.push_back(SLVec3f(  1, 0, 0)); C.push_back(r); v2=i; i++;
    P.push_back(SLVec3f(1-h, w,-w)); C.push_back(r); v3=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    P.push_back(SLVec3f(1-h, w, w)); C.push_back(r); v1=i; i++;
    P.push_back(SLVec3f(1-h,-w, w)); C.push_back(r); v2=i; i++;
    P.push_back(SLVec3f(  1, 0, 0)); C.push_back(r); v3=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    P.push_back(SLVec3f(1-h,-w, w)); C.push_back(r); v1=i; i++;
    P.push_back(SLVec3f(1-h,-w,-w)); C.push_back(r); v2=i; i++;
    P.push_back(SLVec3f(  1, 0, 0)); C.push_back(r); v3=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    P.push_back(SLVec3f(1-h,-w,-w)); C.push_back(r); v1=i; i++;
    P.push_back(SLVec3f(1-h, w,-w)); C.push_back(r); v2=i; i++;
    P.push_back(SLVec3f(  1, 0, 0)); C.push_back(r); v3=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    P.push_back(SLVec3f(1-h, w, w)); C.push_back(r); v1=i; i++;
    P.push_back(SLVec3f(1-h, w,-w)); C.push_back(r); v2=i; i++;
    P.push_back(SLVec3f(1-h,-w,-w)); C.push_back(r); v3=i; i++;
    P.push_back(SLVec3f(1-h,-w, w)); C.push_back(r); v4=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v4);

    P.push_back(SLVec3f( -t, t, t)); C.push_back(r); v1=i; i++;
    P.push_back(SLVec3f( -t, t,-t)); C.push_back(r); v2=i; i++;
    P.push_back(SLVec3f( -t,-t,-t)); C.push_back(r); v3=i; i++;
    P.push_back(SLVec3f( -t,-t, t)); C.push_back(r); v4=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v4);

    // arrow towards +y
    P.push_back(SLVec3f( t,  t, t)); C.push_back(g); v1=i; i++;
    P.push_back(SLVec3f( t,1-h, t)); C.push_back(g); v2=i; i++;
    P.push_back(SLVec3f( t,1-h,-t)); C.push_back(g); v3=i; i++;
    P.push_back(SLVec3f( t,  t,-t)); C.push_back(g); v4=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2);
    I16.push_back(v1); I16.push_back(v4); I16.push_back(v3);

    P.push_back(SLVec3f(-t,  t, t)); C.push_back(g); v1=i; i++;
    P.push_back(SLVec3f(-t,1-h, t)); C.push_back(g); v2=i; i++;
    P.push_back(SLVec3f( t,1-h, t)); C.push_back(g); v3=i; i++;
    P.push_back(SLVec3f( t,  t, t)); C.push_back(g); v4=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2);
    I16.push_back(v1); I16.push_back(v4); I16.push_back(v3);

    P.push_back(SLVec3f(-t,  t,-t)); C.push_back(g); v1=i; i++;
    P.push_back(SLVec3f(-t,1-h,-t)); C.push_back(g); v2=i; i++;
    P.push_back(SLVec3f(-t,1-h, t)); C.push_back(g); v3=i; i++;
    P.push_back(SLVec3f(-t,  t, t)); C.push_back(g); v4=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2);
    I16.push_back(v1); I16.push_back(v4); I16.push_back(v3);

    P.push_back(SLVec3f( t,  t,-t)); C.push_back(g); v1=i; i++;
    P.push_back(SLVec3f( t,1-h,-t)); C.push_back(g); v2=i; i++;
    P.push_back(SLVec3f(-t,1-h,-t)); C.push_back(g); v3=i; i++;
    P.push_back(SLVec3f(-t,  t,-t)); C.push_back(g); v4=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2);
    I16.push_back(v1); I16.push_back(v4); I16.push_back(v3);

    P.push_back(SLVec3f( w,1-h, w)); C.push_back(g); v1=i; i++;
    P.push_back(SLVec3f( 0,  1, 0)); C.push_back(g); v2=i; i++;
    P.push_back(SLVec3f( w,1-h,-w)); C.push_back(g); v3=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2);

    P.push_back(SLVec3f( w,1-h, w)); C.push_back(g); v1=i; i++;
    P.push_back(SLVec3f(-w,1-h, w)); C.push_back(g); v2=i; i++;
    P.push_back(SLVec3f( 0,  1, 0)); C.push_back(g); v3=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2);
    P.push_back(SLVec3f(-w,1-h, w)); C.push_back(g); v1=i; i++;
    P.push_back(SLVec3f(-w,1-h,-w)); C.push_back(g); v2=i; i++;
    P.push_back(SLVec3f( 0,  1, 0)); C.push_back(g); v3=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2);

    P.push_back(SLVec3f(-w,1-h,-w)); C.push_back(g); v1=i; i++;
    P.push_back(SLVec3f( w,1-h,-w)); C.push_back(g); v2=i; i++;
    P.push_back(SLVec3f( 0,  1, 0)); C.push_back(g); v3=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2);

    P.push_back(SLVec3f( w,1-h, w)); C.push_back(g); v1=i; i++;
    P.push_back(SLVec3f( w,1-h,-w)); C.push_back(g); v2=i; i++;
    P.push_back(SLVec3f(-w,1-h,-w)); C.push_back(g); v3=i; i++;
    P.push_back(SLVec3f(-w,1-h, w)); C.push_back(g); v4=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2);
    I16.push_back(v1); I16.push_back(v4); I16.push_back(v3);

    P.push_back(SLVec3f( t, -t, t)); C.push_back(g); v1=i; i++;
    P.push_back(SLVec3f( t, -t,-t)); C.push_back(g); v2=i; i++;
    P.push_back(SLVec3f(-t, -t,-t)); C.push_back(g); v3=i; i++;
    P.push_back(SLVec3f(-t, -t, t)); C.push_back(g); v4=i; i++;
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v2);
    I16.push_back(v1); I16.push_back(v4); I16.push_back(v3);

    // arrow towards +z
    P.push_back(SLVec3f( t, t,  t)); C.push_back(b); v1=i; i++;
    P.push_back(SLVec3f( t, t,1-h)); C.push_back(b); v2=i; i++;
    P.push_back(SLVec3f( t,-t,1-h)); C.push_back(b); v3=i; i++;
    P.push_back(SLVec3f( t,-t,  t)); C.push_back(b); v4=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v4);

    P.push_back(SLVec3f(-t, t,  t)); C.push_back(b); v1=i; i++;
    P.push_back(SLVec3f(-t, t,1-h)); C.push_back(b); v2=i; i++;
    P.push_back(SLVec3f( t, t,1-h)); C.push_back(b); v3=i; i++;
    P.push_back(SLVec3f( t, t,  t)); C.push_back(b); v4=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v4);

    P.push_back(SLVec3f(-t,-t,  t)); C.push_back(b); v1=i; i++;
    P.push_back(SLVec3f(-t,-t,1-h)); C.push_back(b); v2=i; i++;
    P.push_back(SLVec3f(-t, t,1-h)); C.push_back(b); v3=i; i++;
    P.push_back(SLVec3f(-t, t,  t)); C.push_back(b); v4=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v4);

    P.push_back(SLVec3f( t,-t,  t)); C.push_back(b); v1=i; i++;
    P.push_back(SLVec3f( t,-t,1-h)); C.push_back(b); v2=i; i++;
    P.push_back(SLVec3f(-t,-t,1-h)); C.push_back(b); v3=i; i++;
    P.push_back(SLVec3f(-t,-t,  t)); C.push_back(b); v4=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v4);

    P.push_back(SLVec3f( w, w,1-h)); C.push_back(b); v1=i; i++;
    P.push_back(SLVec3f( 0, 0,  1)); C.push_back(b); v2=i; i++;
    P.push_back(SLVec3f( w,-w,1-h)); C.push_back(b); v3=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    P.push_back(SLVec3f( w, w,1-h)); C.push_back(b); v1=i; i++;
    P.push_back(SLVec3f(-w, w,1-h)); C.push_back(b); v2=i; i++;
    P.push_back(SLVec3f( 0, 0,  1)); C.push_back(b); v3=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);

    P.push_back(SLVec3f(-w, w,1-h)); C.push_back(b); v1=i; i++;
    P.push_back(SLVec3f(-w,-w,1-h)); C.push_back(b); v2=i; i++;
    P.push_back(SLVec3f( 0, 0,  1)); C.push_back(b); v3=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    P.push_back(SLVec3f(-w,-w,1-h)); C.push_back(b); v1=i; i++;
    P.push_back(SLVec3f( w,-w,1-h)); C.push_back(b); v2=i; i++;
    P.push_back(SLVec3f( 0, 0,  1)); C.push_back(b); v3=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);

    P.push_back(SLVec3f( w, w,1-h)); C.push_back(b); v1=i; i++;
    P.push_back(SLVec3f( w,-w,1-h)); C.push_back(b); v2=i; i++;
    P.push_back(SLVec3f(-w,-w,1-h)); C.push_back(b); v3=i; i++;
    P.push_back(SLVec3f(-w, w,1-h)); C.push_back(b); v4=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v4);

    P.push_back(SLVec3f( t, t, -t)); C.push_back(b); v1=i; i++;
    P.push_back(SLVec3f( t,-t, -t)); C.push_back(b); v2=i; i++;
    P.push_back(SLVec3f(-t,-t, -t)); C.push_back(b); v3=i; i++;
    P.push_back(SLVec3f(-t, t, -t)); C.push_back(b); v4=i; i++;
    I16.push_back(v1); I16.push_back(v2); I16.push_back(v3);
    I16.push_back(v1); I16.push_back(v3); I16.push_back(v4);
}
//-----------------------------------------------------------------------------

