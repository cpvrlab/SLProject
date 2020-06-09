#pragma once

#include <string>
#include <vector>

#include <Object.h>
#include <Material.h>
#include <SLVec2.h>
#include <SLVec3.h>
#include <SLVec4.h>

using namespace std;
//-----------------------------------------------------------------------------
class Mesh : public Object
{

public:
    Mesh(string name) : Object(name) { ; }

    SLVVec3f  P;   //!< Vector for vertex positions
    SLVVec3f  N;   //!< Vector for vertex normals (opt.)
    SLVVec2f  Tc;  //!< Vector of vertex tex. coords. (opt.)
    SLVCol4f  C;   //!< Vector of vertex colors (opt.)
    SLVuint   I32; //!< Vector of vertex indices 32 bit
    Material* mat; //!< Pointer to the inside material

    //VertexArray vao;    //!< OpenGL Vertex Array Object for drawing
};
//-----------------------------------------------------------------------------
typedef vector<Mesh> VMesh;
//-----------------------------------------------------------------------------
