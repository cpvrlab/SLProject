#pragma once

#include <string>
#include <vector>

#include <Object.h>
#include <Texture.h>
#include <GPUProgram.h>

using namespace std;

//-----------------------------------------------------------------------------
class Material : public Object
{
public:
    Material(string name) : Object(name) { ; }

protected:
    SLCol4f     _ambient;   //!< ambient color (RGB reflection coefficients)
    SLCol4f     _diffuse;   //!< diffuse color (RGB reflection coefficients)
    SLCol4f     _specular;  //!< specular color (RGB reflection coefficients)
    SLCol4f     _emissive;  //!< emissive color coefficients
    SLfloat     _shininess; //!< shininess exponent in Blinn model
    VTexture    _textures;  //!< vector of texture pointers
    GPUProgram* _program;   //!< pointer to a GLSL shader program
};
//-----------------------------------------------------------------------------
typedef vector<Material> VMaterial;
//-----------------------------------------------------------------------------
