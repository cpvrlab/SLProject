#ifndef MATERIAL_H
#define MATERIAL_H

#include <string>
#include <vector>

#include <Object.h>
#include <Texture.h>
#include <GPUProgram.h>
#include <SLVec4.h>

using namespace std;

//-----------------------------------------------------------------------------
class Material : public Object
{
public:
    Material(string name) : Object(name) { ; }

    // Setters
    void setAmbient(const SLCol4f& ambient) { _ambient = ambient; }
    void setDiffuse(const SLCol4f& diffuse) { _diffuse = diffuse; }
    void setSpecular(const SLCol4f& specular) { _specular = specular; }
    void setEmussive(const SLCol4f& emissive) { _emissive = emissive; }
    void setShininess(const SLfloat& shininess) { _shininess = shininess; }
    void setTextures(const VTexture& textures) { _textures = textures; }
    void setProgram(GPUProgram* program) { _program = program; }

    // Getters
    const SLCol4f  ambient() { return _ambient; }
    const SLCol4f  diffuse() { return _diffuse; }
    const SLCol4f  specular() { return _specular; }
    const SLCol4f  emissive() { return _emissive; }
    const SLfloat  shininess() { return _shininess; }
    const VTexture textures() { return _textures; }
    GPUProgram*    program() { return _program; }

    void addTexture(Texture* texture);

protected:
    SLCol4f     _ambient;           //!< ambient color (RGB reflection coefficients)
    SLCol4f     _diffuse;           //!< diffuse color (RGB reflection coefficients)
    SLCol4f     _specular;          //!< specular color (RGB reflection coefficients)
    SLCol4f     _emissive;          //!< emissive color coefficients
    SLfloat     _shininess;         //!< shininess exponent in Blinn model
    VTexture    _textures;          //!< vector of texture pointers
    GPUProgram* _program = nullptr; //!< pointer to a GLSL shader program
};
//-----------------------------------------------------------------------------
typedef vector<Material> VMaterial;
//-----------------------------------------------------------------------------
#endif
