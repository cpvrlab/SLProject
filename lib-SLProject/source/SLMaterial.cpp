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

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <SLApplication.h>
#include <SLMaterial.h>
#include <SLSceneView.h>

//-----------------------------------------------------------------------------
SLfloat SLMaterial::PERFECT = 1000.0f;
//-----------------------------------------------------------------------------
SLMaterial* SLMaterial::current = nullptr;
//-----------------------------------------------------------------------------
SLMaterial* SLMaterial::_defaultGray   = nullptr;
SLMaterial* SLMaterial::_diffuseAttrib = nullptr;
//-----------------------------------------------------------------------------
// Default ctor
SLMaterial::SLMaterial(const SLchar*  name,
                       const SLCol4f& amdi,
                       const SLCol4f& spec,
                       SLfloat        shininess,
                       SLfloat        kr,
                       SLfloat        kt,
                       SLfloat        kn,
                       SLGLProgram*   program) : SLObject(name)
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
    SLApplication::scene->materials().push_back(this);
}
//-----------------------------------------------------------------------------
// Ctor for textures
SLMaterial::SLMaterial(const SLchar* name,
                       SLGLTexture*  texture1,
                       SLGLTexture*  texture2,
                       SLGLTexture*  texture3,
                       SLGLTexture*  texture4,
                       SLGLProgram*  shaderProg) : SLObject(name)
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
    SLApplication::scene->materials().push_back(this);
}
//-----------------------------------------------------------------------------
// Ctor for cone tracer
SLMaterial::SLMaterial(const SLchar* name,
                       SLGLProgram*  shaderProg) : SLObject(name)
{
    _program      = shaderProg;
    _shininess    = 125.0f;
    _roughness    = 0.0f;
    _metalness    = 0.0f;
    _translucency = 0.0f;

    // Add pointer to the global resource vectors for deallocation
    SLApplication::scene->materials().push_back(this);
}

//-----------------------------------------------------------------------------
// Ctor for Cook-Torrance shading
SLMaterial::SLMaterial(const SLchar*  name,
                       const SLCol4f& diffuse,
                       SLfloat        roughness,
                       SLfloat        metalness) : SLObject(name)
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
    _program      = SLApplication::scene->programs(SP_perPixCookTorrance);

    // Add pointer to the global resource vectors for deallocation
    SLApplication::scene->materials().push_back(this);
}
//-----------------------------------------------------------------------------
// Ctor for uniform color material without lighting
SLMaterial::SLMaterial(const SLCol4f& uniformColor, const SLchar* name)
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
    _program      = SLApplication::scene->programs(SP_colorUniform);
    _kr           = 0.0f;
    _kt           = 0.0f;
    _kn           = 1.0f;

    // Add pointer to the global resource vectors for deallocation
    SLApplication::scene->materials().push_back(this);
}
//-----------------------------------------------------------------------------
/*! 
The destructor doesn't delete attached the textures or shader program because
Such shared resources get deleted in the arrays of SLScene.
*/
SLMaterial::~SLMaterial() = default;
//-----------------------------------------------------------------------------
/*!
SLMaterial::activate applies the material parameter to the global render state
and activates the attached shader
*/
void SLMaterial::activate(SLDrawBits drawBits)
{
    SLScene*   s       = SLApplication::scene;
    SLGLState* stateGL = SLGLState::instance();

    // Deactivate shader program of the current active material
    if (current && current->program())
        current->program()->endShader();

    // Set this material as the current material
    current = this;

    // If no shader program is attached add the default shader program
    if (!_program)
    {
        if (!_textures.empty())
        {
            //if (_textures.size() == 1)
                program(s->programs(SP_perVrtBlinnTex));
            //if (_textures.size() > 1 && _textures[1]->texType() == TT_normal)
                //program(s->programs(SP_bumpNormal));
        }
        else
            program(s->programs(SP_perVrtBlinn));
    }

    // Check if shader had compile error and the error texture should be shown
    if (_program && _program->name().find("ErrorTex") != string::npos)
    {
        _textures.clear();
        _textures.push_back(new SLGLTexture("CompileError.png"));
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
    program()->beginUse(this);
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
/*! 
Getter for the global default gray material
*/
SLMaterial* SLMaterial::defaultGray()
{
    if (!_defaultGray)
    {
        _defaultGray = new SLMaterial("default", SLVec4f::GRAY, SLVec4f::WHITE);
        _defaultGray->ambient({0.2f, 0.2f, 0.2f});
    }
    return _defaultGray;
}
//-----------------------------------------------------------------------------
/*! 
The destructor doesn't delete attached the textures or shader program because
Such shared resources get deleted in the arrays of SLScene.
*/
void SLMaterial::defaultGray(SLMaterial* mat)
{
    if (mat == _defaultGray)
        return;

    if (_defaultGray)
    {
        SLVMaterial& list = SLApplication::scene->materials();
        list.erase(remove(list.begin(), list.end(), _defaultGray), list.end());
        delete _defaultGray;
    }

    _defaultGray = mat;
}
//-----------------------------------------------------------------------------
/*! 
Getter for the global diffuse per vertex color attribute material
*/
SLMaterial* SLMaterial::diffuseAttrib()
{
    if (!_diffuseAttrib)
    {
        _diffuseAttrib = new SLMaterial("diffuseAttrib");
        _diffuseAttrib->specular(SLCol4f::BLACK);
        _diffuseAttrib->program(SLApplication::scene->programs(SP_perVrtBlinnColorAttrib));
    }
    return _diffuseAttrib;
}
//-----------------------------------------------------------------------------
/*! 
The destructor doesn't delete attached the textures or shader program because
Such shared resources get deleted in the arrays of SLScene.
*/
void SLMaterial::diffuseAttrib(SLMaterial* mat)
{
    if (mat == _diffuseAttrib)
        return;

    if (_diffuseAttrib)
    {
        SLVMaterial& list = SLApplication::scene->materials();
        list.erase(remove(list.begin(), list.end(), _diffuseAttrib), list.end());
        delete _diffuseAttrib;
    }

    _diffuseAttrib = mat;
}
//-----------------------------------------------------------------------------
