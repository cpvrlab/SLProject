#pragma once

#include <string>
#include <vector>

#include <Object.h>
#include <Material.h>
#include <SLVec2.h>
#include <SLVec3.h>
#include <SLVec4.h>
// #include <SLAABBox.h>

using namespace std;
//-----------------------------------------------------------------------------
class Mesh : public Object
{
public:
    Mesh(string name) : Object(name) { _finalP = &P; }

    void setColor(SLCol4f color);
    // void buildAABB(SLAABBox& aabb, const SLMat4f& wmNode);
    void calcMinMax();

    SLVec3f finalP(SLuint i) { return _finalP->operator[](i); }

    SLVVec3f  P;             //!< Vector for vertex positions
    SLVVec3f  N;             //!< Vector for vertex normals (opt.)
    SLVVec2f  Tc;            //!< Vector of vertex tex. coords. (opt.)
    SLVCol4f  C;             //!< Vector of vertex colors (opt.)
    SLVuint   I32;           //!< Vector of vertex indices 32 bit
    Material* mat = nullptr; //!< Pointer to the outside material

    SLVec3f minP;
    SLVec3f maxP;

protected:
    SLVVec3f* _finalP;

    // VertexArray vao;    //!< OpenGL Vertex Array Object for drawing
};
//-----------------------------------------------------------------------------
typedef vector<Mesh*> VMesh;
//-----------------------------------------------------------------------------
