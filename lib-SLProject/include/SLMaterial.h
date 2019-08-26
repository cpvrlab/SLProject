//#############################################################################
//  File:      SLMaterial.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLMAT_H
#define SLMAT_H

#include <SLDrawBits.h>
#include <SLGLGenericProgram.h>
#include <SLGLProgram.h>
#include <SLGLTexture.h>

class SLSceneView;
class SLNode;

//-----------------------------------------------------------------------------
//! Defines a standard CG material with textures and a shader program
/*!      
The SLMatrial class defines a material with properties for ambient, diffuse, 
specular and emissive light RGBA-reflection. For classic Blinn-Phong shading
the shininess parameter can be used as shininess exponent.
For Cook-Torrance shading the parameters roughness and metallic are provided.
In addition it has coeffitients for reflectivity (kr), transparency (kt) and
refraction index (kn) that can be used in special shaders and ray tracing.
A material can have multiple texture in the std::vector _textures.
The shading has to be implemented in the GLSL program (SLGLProgram) with a
vertex and fragment shader.
All parameters get synced into their corresponding parameter in SLGLState
when a the material gets activated by the method activate.
Light and material paramters get passed to their corresponding shader
variables in SLGLProgram::beginUse.
*/
class SLMaterial : public SLObject
{
    public:
    //! Default ctor
    explicit SLMaterial(const SLchar*  name,
                        const SLCol4f& amdi      = SLCol4f::WHITE,
                        const SLCol4f& spec      = SLCol4f::WHITE,
                        SLfloat        shininess = 100.0f,
                        SLfloat        kr        = 0.0,
                        SLfloat        kt        = 0.0f,
                        SLfloat        kn        = 1.0f);

    //! Ctor for textures
    SLMaterial(const SLchar* name,
               SLGLTexture*  texture1,
               SLGLTexture*  texture2 = nullptr,
               SLGLTexture*  texture3 = nullptr,
               SLGLTexture*  texture4 = nullptr,
               SLGLProgram*  program  = nullptr);

    //! Ctor for Cook-Torrance shading
    SLMaterial(const SLchar*  name,
               const SLCol4f& diffuse,
               SLfloat        roughness,
               SLfloat        metalness);

    //! Ctor for uniform color material without lighting
    explicit SLMaterial(const SLCol4f& uniformColor,
                        const SLchar*  name = (const char*)"Uniform color");

    ~SLMaterial() final;

    //! Sets the material states and passes all variables to the shader program
    void activate(SLDrawBits drawBits);

    //! Returns true if there is any transparency in diffuse alpha or textures
    SLbool hasAlpha() { return (_diffuse.a < 1.0f ||
                                (!_textures.empty() &&
                                 _textures[0]->hasAlpha())); }

//! Returns true if a material has a 3D texture
#ifdef SL_GLES2
    SLbool has3DTexture()
    {
        return false;
    }
#else
    SLbool has3DTexture()
    {
        return !_textures.empty() > 0 &&
               _textures[0]->target() == GL_TEXTURE_3D;
    }
#endif
    //! Returns true if a material with textures tangents as additional attributes
    SLbool needsTangents() { return (_textures.size() >= 2 &&
                                     _textures[0]->target() == GL_TEXTURE_2D &&
                                     _textures[1]->texType() == TT_normal); }
    // Setters
    void ambient(const SLCol4f& ambi) { _ambient = ambi; }
    void diffuse(const SLCol4f& diff) { _diffuse = diff; }
    void ambientDiffuse(const SLCol4f& am_di) { _ambient = _diffuse = am_di; }
    void specular(const SLCol4f& spec) { _specular = spec; }
    void emissive(const SLCol4f& emis) { _emissive = emis; }
    void transmissiv(const SLCol4f& transm) { _transmissive = transm; }
    void translucency(SLfloat transl) { _translucency = transl; }
    void shininess(SLfloat shin)
    {
        if (shin < 0.0f) shin = 0.0;
        _shininess = shin;
    }
    void roughness(SLfloat r) { _roughness = Utils::clamp(r, 0.0f, 1.0f); }
    void metalness(SLfloat m) { _metalness = Utils::clamp(m, 0.0f, 1.0f); }
    void kr(SLfloat kr)
    {
        if (kr < 0.0f) kr = 0.0f;
        if (kr > 1.0f) kr = 1.0f;
        _kr = kr;
    }
    void kt(SLfloat kt)
    {
        if (kt < 0.0f) kt = 0.0f;
        if (kt > 1.0f) kt = 1.0f;
        _kt         = kt;
        _ambient.w  = 1.0f - kt;
        _diffuse.w  = 1.0f - kt;
        _specular.w = 1.0f - kt;
    }
    void kn(SLfloat kn)
    {
        assert(kn >= 0.0f);
        _kn = kn;
    }
    void program(SLGLProgram* sp) { _program = sp; }

    // Getters
    SLCol4f       ambient() { return _ambient; }
    SLCol4f       diffuse() { return _diffuse; }
    SLCol4f       specular() { return _specular; }
    SLCol4f       transmissiv() { return _transmissive; }
    SLCol4f       emissive() { return _emissive; }
    SLfloat       shininess() { return _shininess; }
    SLfloat       roughness() { return _roughness; }
    SLfloat       metalness() { return _metalness; }
    SLfloat       translucency() { return _translucency; }
    SLfloat       kr() { return _kr; }
    SLfloat       kt() { return _kt; }
    SLfloat       kn() { return _kn; }
    SLVGLTexture& textures() { return _textures; }
    SLGLProgram*  program() { return _program; }

    static SLMaterial* defaultGray();
    static void        defaultGray(SLMaterial* mat);
    static SLMaterial* diffuseAttrib();
    static void        diffuseAttrib(SLMaterial* mat);

    // Static variables & functions
    static SLMaterial* current; //!< Current material during scene traversal
    static SLfloat     K;       //!< PM: Constant of gloss calibration (slope of point light at dist 1)
    static SLfloat     PERFECT; //!< PM: shininess/translucency limit

    protected:
    SLCol4f      _ambient;      //!< ambient color (RGB reflection coefficients)
    SLCol4f      _diffuse;      //!< diffuse color (RGB reflection coefficients)
    SLCol4f      _specular;     //!< specular color (RGB reflection coefficients)
    SLCol4f      _transmissive; //!< PM: transmissive color (RGB reflection coefficients)
    SLCol4f      _emissive;     //!< emissive color coefficients
    SLfloat      _shininess;    //!< shininess exponent in Blinn model
    SLfloat      _roughness;    //!< roughness property (0-1) in Cook-Torrance model
    SLfloat      _metalness;    //!< metallic property (0-1) in Cook-Torrance model
    SLfloat      _translucency; //!< PM: translucency exponent for light refraction
    SLfloat      _kr{};         //!< reflection coefficient 0.0 - 1.0
    SLfloat      _kt{};         //!< transmission coefficient 0.0 - 1.0
    SLfloat      _kn{};         //!< refraction index
    SLVGLTexture _textures;     //!< vector of texture pointers
    SLGLProgram* _program{};    //!< pointer to a GLSL shader program

    private:
    static SLMaterial* _defaultGray;   //!< Global default gray color material for meshes that don't define their own.
    static SLMaterial* _diffuseAttrib; //!< Global diffuse reflection material for meshes with color vertex attributes.
};
//-----------------------------------------------------------------------------
//! STL vector of material pointers
typedef std::vector<SLMaterial*> SLVMaterial;
//-----------------------------------------------------------------------------
#endif
