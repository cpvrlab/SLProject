//#############################################################################
//  File:      SLMaterial.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <SLMaterial.h>
#include <SLSceneView.h>
#include <SLGLProgramManager.h>
#include <SLAssetManager.h>

//-----------------------------------------------------------------------------
SLfloat SLMaterial::PERFECT = 1000.0f;
//-----------------------------------------------------------------------------
/*!
 * Default constructor for materials without textures.
 * Materials can be used by multiple meshes (SLMesh). Materials can belong
 * therefore to the global assets such as meshes, materials, textures and
 * shader programs.
 * @param am Pointer to a global asset manager. If passed the asset
 * manager is the owner of the instance and will do the deallocation. If a
 * nullptr is passed the creator is responsible for the deallocation.
 * @param name Name of the material
 * @param amdi Ambient and diffuse color
 * @param spec Specular color
 * @param shininess Shininess exponent (the higher the sharper the gloss)
 * @param kr Reflection coefficient used for ray tracing. (0.0-1.0)
 * @param kt Tranparency coeffitient used for ray tracing. (0.0-1.0)
 * @param kn Refraction index used for ray tracing (1.0-2.5)
 * @param program Pointer to the shader program for the material
 */
SLMaterial::SLMaterial(SLAssetManager* am,
                       const SLchar*   name,
                       const SLCol4f&  amdi,
                       const SLCol4f&  spec,
                       SLfloat         shininess,
                       SLfloat         kr,
                       SLfloat         kt,
                       SLfloat         kn,
                       SLGLProgram*    program,
                       SLstring        compileErrorTexFilePath) : SLObject(name)
{
    _ambient = _diffuse = amdi;
    _specular           = spec;
    _emissive.set(0, 0, 0, 0);
    _shininess       = shininess;
    _roughness       = 0.5f;
    _metalness       = 0.0f;
    _translucency    = 0.0f;
    _getsShadows     = true;
    _shadowBias      = 0.005f;
    _program         = program;

    _kr = kr;
    _kt = kt;
    _kn = kn;

    // sync the transparency coeffitient with the alpha value or vice versa
    if (_kt > 0) _diffuse.w = 1.0f - _kt;
    if (_diffuse.w > 1) _kt = 1.0f - _diffuse.w;

    // Add pointer to the global resource vectors for deallocation
    if (am)
        am->materials().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
 * Constructor for textured materials.
 * Materials can be used by multiple meshes (SLMesh). Materials can belong
 * therefore to the global assets such as meshes, materials, textures and
 * shader programs.
 * @param am Pointer to a global asset manager. If passed the asset
 * manager is the owner of the instance and will do the deallocation. If a
 * nullptr is passed the creator is responsible for the deallocation.
 * @param name Name of the material
 * @param texture1 Texture 1 image filename. If only a filename is
 * passed it will be search on the SLGLTexture::defaultPath.
 * @param texture2 Texture 2 image filename. If only a filename is
 * passed it will be search on the SLGLTexture::defaultPath.
 * @param texture3 Texture 3 image filename. If only a filename is
 * passed it will be search on the SLGLTexture::defaultPath.
 * @param texture4 Texture 4 image filename. If only a filename is
 * passed it will be search on the SLGLTexture::defaultPath.
 * @param shaderProg Pointer to the shader program for the material
 */
SLMaterial::SLMaterial(SLAssetManager* am,
                       const SLchar*   name,
                       SLGLTexture*    texture1,
                       SLGLTexture*    texture2,
                       SLGLTexture*    texture3,
                       SLGLTexture*    texture4,
                       SLGLProgram*    shaderProg,
                       SLstring        compileErrorTexFilePath) : SLObject(name)
{
    _ambient.set(1, 1, 1);
    _diffuse.set(1, 1, 1);
    _specular.set(1, 1, 1);
    _emissive.set(0, 0, 0, 0);
    _shininess       = 125;
    _roughness       = 0.5f;
    _metalness       = 0.0f;
    _translucency    = 0.0f;
    _getsShadows     = true;
    _shadowBias      = 0.0005f;
    _program         = shaderProg;
    _kr              = 0.0f;
    _kt              = 0.0f;
    _kn              = 1.0f;
    _diffuse.w       = 1.0f - _kt;

    if (texture1) _textures.push_back(texture1);
    if (texture2) _textures.push_back(texture2);
    if (texture3) _textures.push_back(texture3);
    if (texture4) _textures.push_back(texture4);

    // Add pointer to the global resource vectors for deallocation
    if (am)
        am->materials().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
 * Constructor for materials used within the cone tracer (SLGLConetracer).
 * Materials can be used by multiple meshes (SLMesh). Materials can belong
 * therefore to the global assets such as meshes, materials, textures and
 * shader programs.
 * @param am Pointer to a global asset manager. If passed the asset
 * manager is the owner of the instance and will do the deallocation. If a
 * nullptr is passed the creator is responsible for the deallocation.
 * @param name Name of the material
 * @param shaderProg Pointer to the shader program for the material.
 */
SLMaterial::SLMaterial(SLAssetManager* am,
                       const SLchar*   name,
                       SLGLProgram*    shaderProg,
                       SLstring        compileErrorTexFilePath) : SLObject(name)
{
    _program         = shaderProg;
    _shininess       = 125.0f;
    _roughness       = 0.0f;
    _metalness       = 0.0f;
    _translucency    = 0.0f;
    _getsShadows     = true;
    _shadowBias      = 0.0005f;

    // Add pointer to the global resource vectors for deallocation
    if (am)
        am->materials().push_back(this);
}

//-----------------------------------------------------------------------------
/*!
 * Constructor for Cook-Torrance shaded materials with roughness and metalness.
 * Materials can be used by multiple meshes (SLMesh). Materials can belong
 * therefore to the global assets such as meshes, materials, textures and
 * shader programs.
 * @param am Pointer to a global asset manager. If passed the asset
 * manager is the owner of the instance and will do the deallocation. If a
 * nullptr is passed the creator is responsible for the deallocation.
 * @param perPixCookTorranceProgram Pointer to the shader program for
 * Cook-Torrance shading
 * @param name Name of the material
 * @param diffuse Diffuse reflection color
 * @param roughness Roughness (0.0-1.0)
 * @param metalness Metalness (0.0-1.0)
 */
SLMaterial::SLMaterial(SLAssetManager* am,
                       SLGLProgram*    perPixCookTorranceProgram,
                       const SLchar*   name,
                       const SLCol4f&  diffuse,
                       SLfloat         roughness,
                       SLfloat         metalness,
                       SLstring        compileErrorTexFilePath) : SLObject(name)
{
    _ambient.set(0, 0, 0); // not used in Cook-Torrance
    _diffuse = diffuse;
    _specular.set(1, 1, 1);                         // not used in Cook-Torrance
    _emissive.set(0, 0, 0, 0);                      // not used in Cook-Torrance
    _shininess       = (1.0f - roughness) * 500.0f; // not used in Cook-Torrance
    _roughness       = roughness;
    _metalness       = metalness;
    _translucency    = 0.0f;
    _getsShadows     = true;
    _shadowBias      = 0.0005f;
    _kr              = 0.0f;
    _kt              = 0.0f;
    _kn              = 1.0f;
    _program         = perPixCookTorranceProgram;

    // Add pointer to the global resource vectors for deallocation
    if (am)
        am->materials().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
 * Constructor for uniform color material without lighting
 * Materials can be used by multiple meshes (SLMesh). Materials can belong
 * therefore to the global assets such as meshes, materials, textures and
 * shader programs.
 * @param am Pointer to a global asset manager. If passed the asset
 * manager is the owner of the instance and will do the deallocation. If a
 * nullptr is passed the creator is responsible for the deallocation.
 * @param colorUniformProgram Pointer to shader program for uniform coloring.
 * @param uniformColor Color to apply
 * @param name Name of the material.
 */
SLMaterial::SLMaterial(SLAssetManager* am,
                       SLGLProgram*    colorUniformProgram,
                       const SLCol4f&  uniformColor,
                       const SLchar*   name,
                       SLstring        compileErrorTexFilePath)
  : SLObject(name)
{
    _ambient.set(0, 0, 0);
    _diffuse = uniformColor;
    _specular.set(0, 0, 0);
    _emissive.set(0, 0, 0, 0);
    _shininess       = 125;
    _roughness       = 0.5f;
    _metalness       = 0.0f;
    _translucency    = 0.0f;
    _program         = colorUniformProgram;
    _kr              = 0.0f;
    _kt              = 0.0f;
    _kn              = 1.0f;
    _getsShadows     = true;
    _shadowBias      = 0.0005f;

    // Add pointer to the global resource vectors for deallocation
    if (am)
        am->materials().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
 * The destructor should be called by the owner of the material. If an asset
 * manager was passed in the constructor it will do it after scene destruction.
 * The textures (SLGLTexture) and the shader program (SLGLProgram) that the
 * material uses will not be deallocated.
*/
SLMaterial::~SLMaterial()
{
    if (_errorTexture)
    {
        delete _errorTexture;
        _errorTexture = nullptr;
    }

    SLGLState* stateGL = SLGLState::instance();
    if (stateGL->currentMaterial() == this)
        stateGL->currentMaterial(nullptr);
}
//-----------------------------------------------------------------------------
/*!
SLMaterial::activate applies the material parameter to the global render state
and activates the attached shader
*/
void SLMaterial::activate(SLDrawBits drawBits, SLVLight* lights)
{
    SLGLState* stateGL = SLGLState::instance();

    if (stateGL->currentMaterial() == this &&
        stateGL->currentMaterial()->program())
        return;

    // Deactivate shader program of the current active material
    if (stateGL->currentMaterial() && stateGL->currentMaterial()->program())
        stateGL->currentMaterial()->program()->endShader();

    // Set this material as the current material
    stateGL->currentMaterial(this);

    // If no shader program is attached add the default shader program
    //todo: this should not happen... then we would not have to do magic
    if (!_program)
    {
        if (!_textures.empty())
            program(SLGLProgramManager::get(SP_perVrtBlinnTex));
        else
            program(SLGLProgramManager::get(SP_perVrtBlinn));
    }

    // Check if shader had compile error and the error texture should be shown
    if (_program && _program->name().find("ErrorTex") != string::npos)
    {
        _textures.clear();
        if (!_errorTexture && !_compileErrorTexFilePath.empty())
            _errorTexture = new SLGLTexture(nullptr, _compileErrorTexFilePath);
        _textures.push_back(_errorTexture);
    }

    // Determine use of shaders & textures
    SLbool useTexture = !drawBits.get(SL_DB_TEXOFF);

    // Enable or disable texturing
    if (useTexture && !_textures.empty())
    {
        for (SLulong i = 0; i < _textures.size(); ++i)
            _textures[i]->bindActive((SLint)i);
    }

    // Activate the shader program now
    program()->beginUse(this, lights);
}
//-----------------------------------------------------------------------------
void SLMaterial::passToUniforms(SLGLProgram* program)
{
    assert(program && "SLMaterial::passToUniforms: No shader program set!");

    SLint loc;
    program->uniform4fv("u_matAmbient", 1, (SLfloat*)&_ambient);
    program->uniform4fv("u_matDiffuse", 1, (SLfloat*)&_diffuse);
    program->uniform4fv("u_matSpecular", 1, (SLfloat*)&_specular);
    program->uniform4fv("u_matEmissive", 1, (SLfloat*)&_emissive);
    program->uniform1f("u_matShininess", _shininess);
    program->uniform1f("u_matRoughness", _roughness);
    program->uniform1f("u_matMetallic", _metalness);
    program->uniform1f("u_matKr", _kr);
    program->uniform1f("u_matKt", _kt);
    program->uniform1f("u_matKn", _kn);
    program->uniform1i("u_matGetsShadows", _getsShadows);
    program->uniform1f("u_matShadowBias", _shadowBias);
    program->uniform1i("u_matHasTexture", !_textures.empty() ? 1 : 0);

    // pass textures
    for (SLint i = 0; i < (SLint)_textures.size(); ++i)
    {
        SLchar name[100];
        sprintf(name, "u_texture%d", i);
        program->uniform1i(name, i);
    }
}
//-----------------------------------------------------------------------------
SLMaterialDiffuseAttribute::SLMaterialDiffuseAttribute()
  : SLMaterial(nullptr, "diffuseAttrib")
{
    specular(SLCol4f::BLACK);
    program(SLGLProgramManager::get(SP_perVrtBlinnColorAttrib));
}
//-----------------------------------------------------------------------------
