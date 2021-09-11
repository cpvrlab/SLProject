//#############################################################################
//  File:      SLMaterial.h
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLMATERIAL_H
#define SLMATERIAL_H

#include <SLDrawBits.h>
#include <SLGLProgramGeneric.h>
#include <SLGLProgram.h>
#include <SLGLTexture.h>
#include <SLNode.h>

class SLSkybox;
class SLSceneView;
class SLAssetManager;

//-----------------------------------------------------------------------------
//! Defines a standard CG material with textures and a shader program
/*!
The SLMaterial class defines a material with properties for ambient, diffuse,
specular and emissive light RGBA-reflection. For classic Blinn-Phong shading
the shininess parameter can be used as shininess exponent.
For Cook-Torrance shading the parameters roughness and metallic are provided.
In addition it has coefficients for reflectivity (kr), transparency (kt) and
refraction index (kn) that can be used in special shaders and ray tracing.
A material can have multiple texture in the vector _textures.
The shading has to be implemented in the GLSL program (SLGLProgram) with a
vertex and fragment shader.
All parameters get synced into their corresponding parameter in SLGLState
when a the material gets activated by the method activate.
Light and material parameters get passed to their corresponding shader
variables in SLGLProgram::beginUse.
*/
class SLMaterial : public SLObject
{
public:
    //! Default ctor for Blinn-Phong light model materials without textures
    explicit SLMaterial(SLAssetManager* am,
                        const SLchar*   name,
                        const SLCol4f&  amdi                    = SLCol4f::WHITE,
                        const SLCol4f&  spec                    = SLCol4f::WHITE,
                        SLfloat         shininess               = 100.0f,
                        SLfloat         kr                      = 0.0,
                        SLfloat         kt                      = 0.0f,
                        SLfloat         kn                      = 1.0f,
                        SLGLProgram*    program                 = nullptr,
                        const SLstring& compileErrorTexFilePath = "");

    //! Ctor for textured Blinn-Phong light model materials
    SLMaterial(SLAssetManager* am,
               const SLchar*   name,
               SLGLTexture*    texture1,
               SLGLTexture*    texture2                = nullptr,
               SLGLTexture*    texture3                = nullptr,
               SLGLTexture*    texture4                = nullptr,
               SLGLProgram*    program                 = nullptr,
               const SLstring& compileErrorTexFilePath = "");

    //! Ctor for Cook-Torrance light model materials
    SLMaterial(SLAssetManager* am,
               SLGLProgram*    perPixCookTorranceProgram,
               const SLchar*   name,
               const SLCol4f&  diffuse,
               SLfloat         roughness,
               SLfloat         metalness,
               const SLstring& compileErrorTexFilePath = "");

    //! Ctor for Cook-Torrance light model materials
    SLMaterial(SLAssetManager* am,
               const SLchar*   name,
               const SLCol4f&  diffuse,
               SLfloat         roughness,
               SLfloat         metalness,
               const SLstring& compileErrorTexFilePath = "");

    //! Ctor for uniform color material without lighting
    explicit SLMaterial(SLAssetManager* am,
                        SLGLProgram*    colorUniformProgram,
                        const SLCol4f&  uniformColor,
                        const SLchar*   name                    = (const char*)"Uniform color",
                        const SLstring& compileErrorTexFilePath = "");

    //! Ctor for cone tracer
    SLMaterial(SLAssetManager* am,
               const SLchar*   name,
               SLGLProgram*    program,
               const SLstring& compileErrorTexFilePath = "");

    //! Ctor for PBR shading with IBL without textures
    SLMaterial(SLAssetManager* am,
               const SLchar*   name,
               SLCol4f         diffuse,
               SLfloat         roughness,
               SLfloat         metalness,
               SLGLProgram*    pbrIblShaderProg,
               SLGLTexture*    irrandianceMap,
               SLGLTexture*    prefilterIrradianceMap,
               SLGLTexture*    brdfLUTTexture);

    //! Ctor for PBR shading with IBL without textures
    SLMaterial(SLAssetManager* am,
               const SLchar*   name,
               SLCol4f         diffuse,
               SLfloat         roughness,
               SLfloat         metalness,
               SLGLTexture*    irrandianceMap,
               SLGLTexture*    prefilterIrradianceMap,
               SLGLTexture*    brdfLUTTexture);

    //! Ctor for PBR material with IBL with PBR textures
    SLMaterial(SLAssetManager* am,
               const SLchar*   name,
               SLGLProgram*    shaderProg,
               SLGLTexture*    texture1,
               SLGLTexture*    texture2,
               SLGLTexture*    texture3,
               SLGLTexture*    texture4,
               SLGLTexture*    texture5,
               SLGLTexture*    texture6,
               SLGLTexture*    texture7,
               SLGLTexture*    texture8);

    //! Ctor for PBR material with IBL with PBR textures
    SLMaterial(SLAssetManager* am,
               const SLchar*   name,
               SLGLTexture*    texture1,
               SLGLTexture*    texture2,
               SLGLTexture*    texture3,
               SLGLTexture*    texture4,
               SLGLTexture*    texture5,
               SLGLTexture*    texture6,
               SLGLTexture*    texture7,
               SLGLTexture*    texture8);

    ~SLMaterial() override;

    void activate(SLCamera* cam,
                  SLVLight* lights,
                  SLSkybox* skybox = nullptr);
    SLint passToUniforms(SLGLProgram* program, SLint nextTexUnit);

    //! Returns true if there is any transparency in diffuse alpha or textures
    SLbool hasAlpha()
    {
        if (_diffuse.a < 1.0)
            return true;

        for (int i = 0; i < _textures[TT_diffuse].size(); i++)
        {
            if (_textures[TT_diffuse][i]->hasAlpha())
                return true;
        }
        return false;
    }

    //! Returns true if a material has a 3D texture
    SLbool has3DTexture()
    {
        return !_textures3d.empty();
    }
    SLbool needsTangents()
    {
        return !_textures[TT_normal].empty() &&
               _textures[TT_normal][0]->target() == GL_TEXTURE_2D;
    }
    SLbool hasTextureType(SLTextureType tt)
    {
        return !_textures[tt].empty();
    }
    void removeTextureType(SLTextureType tt)
    {
        _numTextures -= _textures[tt].size();
        _textures[tt].clear();
    }
    void addTexture(SLGLTexture* texture);

    // Setters
    void ambient(const SLCol4f& ambi) { _ambient = ambi; }
    void diffuse(const SLCol4f& diff) { _diffuse = diff; }
    void ambientDiffuse(const SLCol4f& am_di) { _ambient = _diffuse = am_di; }
    void specular(const SLCol4f& spec) { _specular = spec; }
    void emissive(const SLCol4f& emis) { _emissive = emis; }
    void transmissive(const SLCol4f& transm) { _transmissive = transm; }
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
    void getsShadows(SLbool receivesShadows) { _getsShadows = receivesShadows; }
    void program(SLGLProgram* sp) { _program = sp; }

    // Getters
    SLAssetManager* assetManager() { return _assetManager; }
    SLLightModel    lightModel() { return _lightModel; }
    SLCol4f         ambient() { return _ambient; }
    SLCol4f         diffuse() { return _diffuse; }
    SLCol4f         specular() { return _specular; }
    SLCol4f         transmissive() { return _transmissive; }
    SLCol4f         emissive() { return _emissive; }
    SLfloat         shininess() const { return _shininess; }
    SLfloat         roughness() const { return _roughness; }
    SLfloat         metalness() const { return _metalness; }
    SLfloat         translucency() const { return _translucency; }
    SLfloat         kr() const { return _kr; }
    SLfloat         kt() const { return _kt; }
    SLfloat         kn() const { return _kn; }
    SLbool          getsShadows() const { return _getsShadows; }
    SLuint          numTextures() { return _numTextures; }
    SLGLProgram*    program() { return _program; }
    SLVNode&        nodesVisible2D() { return _nodesVisible2D; }
    SLVNode&        nodesVisible3D() { return _nodesVisible3D; }

    SLVGLTexture& textures(SLTextureType type) { return _textures[type]; }
    SLVGLTexture& textures3d() { return _textures3d; }

    // Setters
    void assetManager(SLAssetManager* am) { _assetManager = am; }
    void lightModel(SLLightModel lm) { _lightModel = lm; }

    // Static variables & functions
    static SLfloat K;       //!< PM: Constant of gloss calibration (slope of point light at dist 1)
    static SLfloat PERFECT; //!< PM: shininess/translucency limit

protected:
    SLAssetManager* _assetManager; //!< Pointer to the asset manager (the owner) if available
    SLLightModel    _lightModel;   //!< Enumeration for lighting model (LM_BlinnPhong or LM_CookTorrance)
    SLCol4f         _ambient;      //!< ambient color (RGB reflection coefficients)
    SLCol4f         _diffuse;      //!< diffuse color (RGB reflection coefficients)
    SLCol4f         _specular;     //!< specular color (RGB reflection coefficients)
    SLCol4f         _transmissive; //!< PM: transmissive color (RGB reflection coefficients)
    SLCol4f         _emissive;     //!< emissive color coefficients
    SLfloat         _shininess;    //!< shininess exponent in Blinn-Phong model
    SLfloat         _roughness;    //!< roughness property (0-1) in Cook-Torrance model
    SLfloat         _metalness;    //!< metallic property (0-1) in Cook-Torrance model
    SLfloat         _translucency; //!< PM: translucency exponent for light refraction
    SLfloat         _kr{};         //!< reflection coefficient 0.0 - 1.0
    SLfloat         _kt{};         //!< transmission coefficient 0.0 - 1.0
    SLfloat         _kn{};         //!< refraction index
    SLbool          _getsShadows;  //!< true if shadows are visible on this material
    SLGLProgram*    _program{};    //!< pointer to a GLSL shader program
    SLint           _numTextures;  //!< Number of textures in all _textures vectors array

    SLVGLTexture _textures[TT_numTextureType]; //!< Array of texture vectors one for each type
    SLVGLTexture _textures3d;                  //!< Texture vector for diffuse 3D textures
    SLGLTexture* _errorTexture = nullptr;      //!< Pointer to error texture that is shown if another texture fails
    SLstring     _compileErrorTexFilePath;     //!< Path to the error texture

    SLVNode _nodesVisible2D; //!< Vector of all visible 2D nodes of with this material
    SLVNode _nodesVisible3D; //!< Vector of all visible 3D nodes of with this material
};
//-----------------------------------------------------------------------------
//! STL vector of material pointers
typedef vector<SLMaterial*> SLVMaterial;
//-----------------------------------------------------------------------------
//! Global default gray color material for meshes that don't define their own.
/*!
 * Because the default material depends a default shader program
 * (SLGLDefaultProgPerVrtBlinn or SLGLDefaultProgPerVrtBlinnTm) that itself
 * depends on the scene configuration (e.g. the num. of lights) ist MUST be
 * deleted at scene destruction.
 */
class SLMaterialDefaultGray : public SLMaterial
{
public:
    static SLMaterialDefaultGray* instance()
    {
        if (!_instance)
            _instance = new SLMaterialDefaultGray;
        return _instance;
    }
    static void deleteInstance()
    {
        if (_instance)
        {
            delete _instance;
            _instance = nullptr;
        }
    }

private:
    SLMaterialDefaultGray()
      : SLMaterial(nullptr, "default", SLVec4f::GRAY, SLVec4f::WHITE)
    {
        ambient({0.2f, 0.2f, 0.2f});
    }

    static SLMaterialDefaultGray* _instance;
};
//-----------------------------------------------------------------------------
//! Global default color attribute material for meshes that have colors per vertex
class SLMaterialDefaultColorAttribute : public SLMaterial
{
public:
    static SLMaterialDefaultColorAttribute* instance()
    {
        if (!_instance)
            _instance = new SLMaterialDefaultColorAttribute;
        return _instance;
    }
    static void deleteInstance()
    {
        if (_instance)
        {
            delete _instance;
            _instance = nullptr;
        }
    }

private:
    SLMaterialDefaultColorAttribute()
      : SLMaterial(nullptr, "ColorAttribute")
    {
        program(SLGLProgramManager::get(SP_colorAttribute));
    }

    static SLMaterialDefaultColorAttribute* _instance;
};
//-----------------------------------------------------------------------------
#endif
