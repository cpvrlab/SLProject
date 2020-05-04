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
// Default ctor
SLMaterial::SLMaterial(SLAssetManager* s,
                       const SLchar*   name,
                       const SLCol4f&  amdi,
                       const SLCol4f&  spec,
                       SLfloat         shininess,
                       SLfloat         kr,
                       SLfloat         kt,
                       SLfloat         kn,
                       SLGLProgram*    program) : SLObject(name)
{
    _ambient = _diffuse = amdi;
    _specular           = spec;
    _emissive.set(0, 0, 0, 0);
    _shininess    = shininess;
    _roughness    = 0.5f;
    _metalness    = 0.0f;
    _translucency = 0.0f;
    _program      = program;

    _kr = kr;
    _kt = kt;
    _kn = kn;

    // sync the transparency coeffitient with the alpha value or vice versa
    if (_kt > 0) _diffuse.w = 1.0f - _kt;
    if (_diffuse.w > 1) _kt = 1.0f - _diffuse.w;

    // Add pointer to the global resource vectors for deallocation
    if (s)
        s->materials().push_back(this);
}
//-----------------------------------------------------------------------------
// Ctor for textures
SLMaterial::SLMaterial(SLAssetManager* s,
                       const SLchar*   name,
                       SLGLTexture*    texture1,
                       SLGLTexture*    texture2,
                       SLGLTexture*    texture3,
                       SLGLTexture*    texture4,
                       SLGLProgram*    shaderProg) : SLObject(name)
{
    _ambient.set(1, 1, 1);
    _diffuse.set(1, 1, 1);
    _specular.set(1, 1, 1);
    _emissive.set(0, 0, 0, 0);
    _shininess    = 125;
    _roughness    = 0.5f;
    _metalness    = 0.0f;
    _translucency = 0.0f;
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
    if (s)
        s->materials().push_back(this);
}
//-----------------------------------------------------------------------------
// Ctor for cone tracer
SLMaterial::SLMaterial(SLAssetManager* s,
                       const SLchar*   name,
                       SLGLProgram*    shaderProg) : SLObject(name)
{
    _program      = shaderProg;
    _shininess    = 125.0f;
    _roughness    = 0.0f;
    _metalness    = 0.0f;
    _translucency = 0.0f;

    // Add pointer to the global resource vectors for deallocation
    if (s)
        s->materials().push_back(this);
}

//-----------------------------------------------------------------------------
// Ctor for Cook-Torrance shading
SLMaterial::SLMaterial(SLAssetManager* s,
                       SLGLProgram*    perPixCookTorranceProgram,
                       const SLchar*   name,
                       const SLCol4f&  diffuse,
                       SLfloat         roughness,
                       SLfloat         metalness) : SLObject(name)
{
    _ambient.set(0, 0, 0); // not used in Cook-Torrance
    _diffuse = diffuse;
    _specular.set(1, 1, 1);                      // not used in Cook-Torrance
    _emissive.set(0, 0, 0, 0);                   // not used in Cook-Torrance
    _shininess    = (1.0f - roughness) * 500.0f; // not used in Cook-Torrance
    _roughness    = roughness;
    _metalness    = metalness;
    _translucency = 0.0f;
    _kr           = 0.0f;
    _kt           = 0.0f;
    _kn           = 1.0f;
    _program      = perPixCookTorranceProgram;

    // Add pointer to the global resource vectors for deallocation
    if (s)
        s->materials().push_back(this);
}
//-----------------------------------------------------------------------------
// Ctor for uniform color material without lighting
SLMaterial::SLMaterial(SLAssetManager* s,
                       SLGLProgram*    colorUniformProgram,
                       const SLCol4f&  uniformColor,
                       const SLchar*   name)
  : SLObject(name)
{
    _ambient.set(0, 0, 0);
    _diffuse = uniformColor;
    _specular.set(0, 0, 0);
    _emissive.set(0, 0, 0, 0);
    _shininess    = 125;
    _roughness    = 0.5f;
    _metalness    = 0.0f;
    _translucency = 0.0f;
    _program      = colorUniformProgram;
    //_program      = s->programs(SP_colorUniform);
    _kr = 0.0f;
    _kt = 0.0f;
    _kn = 1.0f;

    // Add pointer to the global resource vectors for deallocation
    if (s)
        s->materials().push_back(this);
}
//-----------------------------------------------------------------------------
/*! 
The destructor doesn't delete attached the textures or shader program because
Such shared resources get deleted in the arrays of SLScene.
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
    {
        stateGL->currentMaterial(nullptr);
    }
}
//-----------------------------------------------------------------------------
/*!
SLMaterial::activate applies the material parameter to the global render state
and activates the attached shader
*/
void SLMaterial::activate(SLDrawBits     drawBits,
                          const SLCol4f& globalAmbiLight)
{
    SLGLState* stateGL = SLGLState::instance();

    if (stateGL->currentMaterial() == this && stateGL->currentMaterial()->program())
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
        {
            //if (_textures.size() == 1)
            program(SLGLProgramManager::get(SP_perVrtBlinnTex));
            //if (_textures.size() > 1 && _textures[1]->texType() == TT_normal)
            //program(s->programs(SP_bumpNormal));
        }
        else
            program(SLGLProgramManager::get(SP_perVrtBlinn));
    }

    // Check if shader had compile error and the error texture should be shown
    if (_program && _program->name().find("ErrorTex") != string::npos)
    {
        _textures.clear();
        _errorTexture = new SLGLTexture(nullptr, "CompileError.png");
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
    program()->beginUse(this, globalAmbiLight);
}
//-----------------------------------------------------------------------------
void SLMaterial::passToUniforms(SLGLProgram* program)
{
    assert(program && "SLMaterial::passToUniforms: No shader program set!");

    SLint loc;
    loc = program->uniform4fv("u_matAmbient", 1, (SLfloat*)&_ambient);
    loc = program->uniform4fv("u_matDiffuse", 1, (SLfloat*)&_diffuse);
    loc = program->uniform4fv("u_matSpecular", 1, (SLfloat*)&_specular);
    loc = program->uniform4fv("u_matEmissive", 1, (SLfloat*)&_emissive);
    loc = program->uniform1f("u_matShininess", _shininess);
    loc = program->uniform1f("u_matRoughness", _roughness);
    loc = program->uniform1f("u_matMetallic", _metalness);
    loc = program->uniform1f("u_matKr", _kr);
    loc = program->uniform1f("u_matKt", _kt);
    loc = program->uniform1f("u_matKn", _kn);
    loc = program->uniform1i("u_matHasTexture", !_textures.empty() ? 1 : 0);
}
//-----------------------------------------------------------------------------
SLMaterialDiffuseAttribute::SLMaterialDiffuseAttribute()
  : SLMaterial(nullptr, "diffuseAttrib")
{
    specular(SLCol4f::BLACK);
    program(SLGLProgramManager::get(SP_perVrtBlinnColorAttrib));
}
