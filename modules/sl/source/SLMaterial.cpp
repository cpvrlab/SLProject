//#############################################################################
//  File:      SLMaterial.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLMaterial.h>
#include <SLSceneView.h>
#include <SLAssetManager.h>
#include <SLGLProgramGenerated.h>

//-----------------------------------------------------------------------------
SLfloat SLMaterial::PERFECT = 1000.0f;
//-----------------------------------------------------------------------------
/*!
 Default constructor for Blinn-Phong light model materials without textures.
 Materials can be used by multiple meshes (SLMesh). Materials can belong
 therefore to the global assets such as meshes, materials, textures and
 shader programs.
 @param am Pointer to a global asset manager. If passed the asset
 manager is the owner of the instance and will do the deallocation. If a
 nullptr is passed the creator is responsible for the deallocation.
 @param name Name of the material
 @param amdi Ambient and diffuse color
 @param spec Specular color
 @param shininess Shininess exponent (the higher the sharper the gloss)
 @param kr Reflection coefficient used for ray tracing. (0.0-1.0)
 @param kt Transparency coefficient used for ray tracing. (0.0-1.0)
 @param kn Refraction index used for ray tracing (1.0-2.5)
 @param program Pointer to the shader program for the material
 @param compileErrorTexFilePath Path to an error texture
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
                       const SLstring& compileErrorTexFilePath) : SLObject(name)
{
    _assetManager = am;
    _lightModel   = LM_BlinnPhong;
    _ambient = _diffuse = amdi;
    _specular           = spec;
    _emissive.set(0, 0, 0, 0);
    _shininess    = shininess;
    _roughness    = 0.5f;
    _metalness    = 0.0f;
    _translucency = 0.0f;
    _getsShadows  = true;
    _program      = program;

    _kr = kr;
    _kt = kt;
    _kn = kn;

    // sync the transparency coefficient with the alpha value or vice versa
    if (_kt > 0) _diffuse.w = 1.0f - _kt;
    if (_diffuse.w > 1) _kt = 1.0f - _diffuse.w;

    // Add pointer to the global resource vectors for deallocation
    if (am)
        am->materials().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
 Constructor for textured Blinn-Phong light model materials.
 Materials can be used by multiple meshes (SLMesh). Materials can belong
 therefore to the global assets such as meshes, materials, textures and
 shader programs.
 @param am Pointer to a global asset manager. If passed the asset
 manager is the owner of the instance and will do the deallocation. If a
 nullptr is passed the creator is responsible for the deallocation.
 @param name Name of the material
 @param texture1 Texture 1 image filename. If only a filename is
 passed it will be search on the SLGLTexture::defaultPath.
 @param texture2 Texture 2 image filename. If only a filename is
 passed it will be search on the SLGLTexture::defaultPath.
 @param texture3 Texture 3 image filename. If only a filename is
 passed it will be search on the SLGLTexture::defaultPath.
 @param texture4 Texture 4 image filename. If only a filename is
 passed it will be search on the SLGLTexture::defaultPath.
 @param shaderProg Pointer to the shader program for the material
 @param compileErrorTexFilePath Path to an error texture
 */
SLMaterial::SLMaterial(SLAssetManager* am,
                       const SLchar*   name,
                       SLGLTexture*    texture1,
                       SLGLTexture*    texture2,
                       SLGLTexture*    texture3,
                       SLGLTexture*    texture4,
                       SLGLProgram*    shaderProg,
                       const SLstring& compileErrorTexFilePath) : SLObject(name)
{
    _assetManager = am;
    _lightModel   = LM_BlinnPhong;
    _ambient.set(1, 1, 1);
    _diffuse.set(1, 1, 1);
    _specular.set(1, 1, 1);
    _emissive.set(0, 0, 0, 0);
    _shininess    = 125;
    _roughness    = 0.5f;
    _metalness    = 0.0f;
    _translucency = 0.0f;
    _getsShadows  = true;
    _program      = shaderProg;
    _kr           = 0.0f;
    _kt           = 0.0f;
    _kn           = 1.0f;
    _diffuse.w    = 1.0f - _kt;

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
 Constructor for materials used within the cone tracer (SLGLConetracer).
 Materials can be used by multiple meshes (SLMesh). Materials can belong
 therefore to the global assets such as meshes, materials, textures and
 shader programs.
 @param am Pointer to a global asset manager. If passed the asset
 manager is the owner of the instance and will do the deallocation. If a
 nullptr is passed the creator is responsible for the deallocation.
 @param name Name of the material
 @param shaderProg Pointer to the shader program for the material.
 @param compileErrorTexFilePath Path to an error texture
 */
SLMaterial::SLMaterial(SLAssetManager* am,
                       const SLchar*   name,
                       SLGLProgram*    shaderProg,
                       const SLstring& compileErrorTexFilePath) : SLObject(name)
{
    _assetManager = am;
    _program      = shaderProg;
    _lightModel   = LM_BlinnPhong;
    _ambient.set(1, 1, 1);
    _diffuse.set(1, 1, 1);
    _specular.set(1, 1, 1);
    _emissive.set(0, 0, 0, 0);
    _shininess    = 125.0f;
    _roughness    = 0.0f;
    _metalness    = 0.0f;
    _translucency = 0.0f;
    _getsShadows  = true;

    // Add pointer to the global resource vectors for deallocation
    if (am)
        am->materials().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
 Constructor for Cook-Torrance shaded materials with roughness and metalness.
 Materials can be used by multiple meshes (SLMesh). Materials can belong
 therefore to the global assets such as meshes, materials, textures and
 shader programs.
 @param am Pointer to a global asset manager. If passed the asset
 manager is the owner of the instance and will do the deallocation. If a
 nullptr is passed the creator is responsible for the deallocation.
 @param perPixCookTorranceProgram Pointer to the shader program for
 Cook-Torrance shading
 @param name Name of the material
 @param diffuse Diffuse reflection color
 @param roughness Roughness (0.0-1.0)
 @param metalness Metalness (0.0-1.0)
 @param compileErrorTexFilePath Path to an error texture
 */
SLMaterial::SLMaterial(SLAssetManager* am,
                       SLGLProgram*    perPixCookTorranceProgram,
                       const SLchar*   name,
                       const SLCol4f&  diffuse,
                       SLfloat         roughness,
                       SLfloat         metalness,
                       const SLstring& compileErrorTexFilePath) : SLObject(name)
{
    _assetManager = am;
    _lightModel   = LM_CookTorrance;
    _ambient.set(0, 0, 0); // not used in Cook-Torrance
    _diffuse = diffuse;
    _specular.set(1, 1, 1);                      // not used in Cook-Torrance
    _emissive.set(0, 0, 0, 0);                   // not used in Cook-Torrance
    _shininess    = (1.0f - roughness) * 500.0f; // not used in Cook-Torrance
    _roughness    = roughness;
    _metalness    = metalness;
    _translucency = 0.0f;
    _getsShadows  = true;
    _kr           = 0.0f;
    _kt           = 0.0f;
    _kn           = 1.0f;
    _program      = perPixCookTorranceProgram;

    // Add pointer to the global resource vectors for deallocation
    if (am)
        am->materials().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
 Constructor for uniform color material without lighting
 Materials can be used by multiple meshes (SLMesh). Materials can belong
 therefore to the global assets such as meshes, materials, textures and
 shader programs.
 @param am Pointer to a global asset manager. If passed the asset
 manager is the owner of the instance and will do the deallocation. If a
 nullptr is passed the creator is responsible for the deallocation.
 @param colorUniformProgram Pointer to shader program for uniform coloring.
 @param uniformColor Color to apply
 @param name Name of the material
 @param compileErrorTexFilePath Path to an error texture
 */
SLMaterial::SLMaterial(SLAssetManager* am,
                       SLGLProgram*    colorUniformProgram,
                       const SLCol4f&  uniformColor,
                       const SLchar*   name,
                       const SLstring& compileErrorTexFilePath)
  : SLObject(name)
{
    _assetManager = am;
    _lightModel   = LM_Custom;
    _ambient.set(0, 0, 0);
    _diffuse = uniformColor;
    _specular.set(0, 0, 0);
    _emissive.set(0, 0, 0, 0);
    _shininess    = 125;
    _roughness    = 0.5f;
    _metalness    = 0.0f;
    _translucency = 0.0f;
    _program      = colorUniformProgram;
    _kr           = 0.0f;
    _kt           = 0.0f;
    _kn           = 1.0f;
    _getsShadows  = true;

    // Add pointer to the global resource vectors for deallocation
    if (am)
        am->materials().push_back(this);
}

//-----------------------------------------------------------------------------
    //! Ctor for PBR shading with IBL without textures
SLMaterial::SLMaterial(SLAssetManager* am,
                       const SLchar*   name,
                       SLCol4f         diffuse,
                       SLfloat         roughness,
                       SLfloat         metalness,
                       SLGLProgram*    pbrIblShaderProg,
                       SLGLTexture*    irrandianceMap,
                       SLGLTexture*    prefilterIrradianceMap,
                       SLGLTexture*    brdfLUTTexture)
{
    _ambient.set(0, 0, 0); // not used in Cook-Torrance
    _diffuse = diffuse;
    _specular.set(1, 1, 1);                   // not used in Cook-Torrance
    _emissive.set(0, 0, 0, 0);                // not used in Cook-Torrance
    _shininess = (1.0f - roughness) * 500.0f; // not used in Cook-Torrance
    _roughness = roughness;
    _metalness = metalness;

    _kr = 0.0f;
    _kt = 0.0f;
    _kn = 1.0f;

    _program = pbrIblShaderProg;

    if (irrandianceMap) _textures.push_back(irrandianceMap);
    if (prefilterIrradianceMap) _textures.push_back(prefilterIrradianceMap);
    if (brdfLUTTexture) _textures.push_back(brdfLUTTexture);

    // Add pointer to the global resource vectors for deallocation
    if (am)
        am->materials().push_back(this);
}
//-----------------------------------------------------------------------------
// Ctor for textures with PBR materials
SLMaterial::SLMaterial(SLAssetManager* am,
                       const SLchar*   name,
                       SLGLProgram*    shaderProg,
                       SLGLTexture*    texture1,
                       SLGLTexture*    texture2,
                       SLGLTexture*    texture3,
                       SLGLTexture*    texture4,
                       SLGLTexture*    texture5,
                       SLGLTexture*    texture6,
                       SLGLTexture*    texture7,
                       SLGLTexture*    texture8)
{
    _ambient.set(1, 1, 1);
    _diffuse.set(1, 1, 1);
    _specular.set(1, 1, 1);
    _emissive.set(0, 0, 0, 0);
    _shininess = 125;
    _roughness = 0.5f;
    _metalness = 0.0f;

    if (texture1) _textures.push_back(texture1);
    if (texture2) _textures.push_back(texture2);
    if (texture3) _textures.push_back(texture3);
    if (texture4) _textures.push_back(texture4);
    if (texture5) _textures.push_back(texture5);
    if (texture6) _textures.push_back(texture6);
    if (texture7) _textures.push_back(texture7);
    if (texture8) _textures.push_back(texture8);

    _program = shaderProg;

    _kr        = 0.0f;
    _kt        = 0.0f;
    _kn        = 1.0f;
    _diffuse.w = 1.0f - _kt;

    // Add pointer to the global resource vectors for deallocation
    if (am)
        am->materials().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
 The destructor should be called by the owner of the material. If an asset
 manager was passed in the constructor it will do it after scene destruction.
 The textures (SLGLTexture) and the shader program (SLGLProgram) that the
 material uses will not be deallocated.
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
 SLMaterial::activate activates this material for rendering if it is not yet
 the active one and set as SLGLState::currentMaterial. If this material has
 not yet a shader program assigned (SLMaterial::_program) a suitable program
 will be generated with an instance of SLGLProgramGenerated.
 At the end the shader program will begin its usage with SLGLProgram::beginUse.
*/
void SLMaterial::activate(SLCamera* cam, SLVLight* lights)
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

    // If no shader program is attached add a generated shader program
    // A 3D object can be stored without material or shader program information.
    if (!_program)
    {
        // Check first the asset manager if the requested program type already exists
        string programName;
        SLGLProgramGenerated::buildProgramName(this, lights, programName);
        _program = _assetManager->getProgramByName(programName);

        // If the program was not found by name generate a new one
        if (!_program)
            _program = new SLGLProgramGenerated(_assetManager, programName, this, lights);
    }

    // Check if shader had compile error and the error texture should be shown
    if (_program && _program->name().find("ErrorTex") != string::npos)
    {
        _textures.clear();
        if (!_errorTexture && !_compileErrorTexFilePath.empty())
            _errorTexture = new SLGLTexture(nullptr, _compileErrorTexFilePath);
        _textures.push_back(_errorTexture);
    }

    // Activate the shader program now
    _program->beginUse(cam, this, lights);
}
//-----------------------------------------------------------------------------
//! Passes all material parameters as uniforms to the passed shader program
void SLMaterial::passToUniforms(SLGLProgram* program)
{
    assert(program && "SLMaterial::passToUniforms: No shader program set!");

    program->uniform4fv("u_matAmbi", 1, (SLfloat*)&_ambient);
    program->uniform4fv("u_matDiff", 1, (SLfloat*)&_diffuse);
    program->uniform4fv("u_matSpec", 1, (SLfloat*)&_specular);
    program->uniform4fv("u_matEmis", 1, (SLfloat*)&_emissive);
    program->uniform1f("u_matShin", _shininess);
    program->uniform1f("u_matRough", _roughness);
    program->uniform1f("u_matMetal", _metalness);
    program->uniform1f("u_matKr", _kr);
    program->uniform1f("u_matKt", _kt);
    program->uniform1f("u_matKn", _kn);
    program->uniform1i("u_matGetsShadows", _getsShadows);
    program->uniform1i("u_matHasTexture", !_textures.empty() ? 1 : 0);

    // pass textures unit id to the sampler uniform
    for (SLuint texUnit = 0; texUnit < _textures.size(); ++texUnit)
    {
        SLchar name[100];
        _textures[texUnit]->bindActive(texUnit);
        sprintf(name, "u_matTexture%d", texUnit);
        program->uniform1i(name, texUnit);
    }
}
//-----------------------------------------------------------------------------
