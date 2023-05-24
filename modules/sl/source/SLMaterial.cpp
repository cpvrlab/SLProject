//#############################################################################
//  File:      SLMaterial.cpp
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLMaterial.h>
#include <SLSceneView.h>
#include <SLAssetManager.h>
#include <SLGLProgramGenerated.h>
#include <SLSkybox.h>

//-----------------------------------------------------------------------------
SLfloat SLMaterial::PERFECT = 1000.0f;
//-----------------------------------------------------------------------------
/*!
 Default constructor for Blinn-Phong reflection model materials without textures.
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
 @param program Pointer to the shader program for the material.
 If none is passed a program will be generated from the passed parameters.
*/
SLMaterial::SLMaterial(SLAssetManager* am,
                       const SLchar*   name,
                       const SLCol4f&  amdi,
                       const SLCol4f&  spec,
                       SLfloat         shininess,
                       SLfloat         kr,
                       SLfloat         kt,
                       SLfloat         kn,
                       SLGLProgram*    program) : SLObject(name)
{
    _assetManager    = am;
    _reflectionModel = RM_BlinnPhong;
    _ambient = _diffuse = amdi;
    _specular           = spec;
    _emissive.set(0, 0, 0, 0);
    _shininess    = shininess;
    _roughness    = 0.5f;
    _metalness    = 0.0f;
    _translucency = 0.0f;
    _getsShadows  = true;
    _program      = program;
    _skybox       = nullptr;
    _numTextures  = 0;

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
/*! Constructor for textured Blinn-Phong reflection model materials.
 Materials can be used by multiple meshes (SLMesh). Materials can belong
 therefore to the global assets such as meshes, materials, textures and
 shader programs.
 @param am Pointer to a global asset manager. If passed the asset
 manager is the owner of the instance and will do the deallocation. If a
 nullptr is passed the creator is responsible for the deallocation.
 @param name Name of the material
 @param texture1 Pointer to an SLGLTexture of a specific SLTextureType
 @param texture2 Pointer to an SLGLTexture of a specific SLTextureType
 @param texture3 Pointer to an SLGLTexture of a specific SLTextureType
 @param texture4 Pointer to an SLGLTexture of a specific SLTextureType
 @param program Pointer to the shader program for the material.
 If none is passed a program will be generated from the passed parameters.
 */
SLMaterial::SLMaterial(SLAssetManager* am,
                       const SLchar*   name,
                       SLGLTexture*    texture1,
                       SLGLTexture*    texture2,
                       SLGLTexture*    texture3,
                       SLGLTexture*    texture4,
                       SLGLProgram*    program) : SLObject(name)
{
    _assetManager    = am;
    _reflectionModel = RM_BlinnPhong;
    _ambient.set(1, 1, 1);
    _diffuse.set(1, 1, 1);
    _specular.set(1, 1, 1);
    _emissive.set(0, 0, 0, 0);
    _shininess    = 125;
    _roughness    = 0.5f;
    _metalness    = 0.0f;
    _translucency = 0.0f;
    _getsShadows  = true;
    _kr           = 0.0f;
    _kt           = 0.0f;
    _kn           = 1.0f;
    _diffuse.w    = 1.0f - _kt;
    _skybox       = nullptr;
    _program      = program;

    _numTextures = 0;
    addTexture(texture1);
    addTexture(texture2);
    addTexture(texture3);
    addTexture(texture4);

    if (_textures[TT_roughness].size() > 0 || _textures[TT_metallic].size() > 0)
        _reflectionModel = RM_CookTorrance;

    // Add pointer to the global resource vectors for deallocation
    if (am)
        am->materials().push_back(this);
}
//-----------------------------------------------------------------------------
/*! Constructor for Cook-Torrance shaded materials with roughness and metalness.
 Materials can be used by multiple meshes (SLMesh). Materials can belong
 therefore to the global assets such as meshes, materials, textures and
 shader programs.
 @param am Pointer to a global asset manager. If passed the asset manager
 is the owner of the instance and will do the deallocation. If a nullptr
 is passed the creator is responsible for the deallocation.
 @param name Name of the material
 @param skybox Pointer to the skybox if available
 @param diffuse Diffuse reflection color
 @param roughness Roughness (0.0-1.0)
 @param metalness Metalness (0.0-1.0)
 @param program Pointer to the shader program for the material.
 If none is passed a program will be generated from the passed parameters.
 */
SLMaterial::SLMaterial(SLAssetManager* am,
                       const SLchar*   name,
                       SLSkybox*       skybox,
                       SLCol4f         diffuse,
                       SLfloat         roughness,
                       SLfloat         metalness,
                       SLGLProgram*    program) : SLObject(name)
{
    _assetManager = am;
    _ambient.set(0, 0, 0);                          // not used in Cook-Torrance
    _diffuse = diffuse;
    _specular.set(1, 1, 1);                         // not used in Cook-Torrance
    _emissive.set(0, 0, 0, 0);                      // not used in Cook-Torrance
    _shininess       = (1.0f - roughness) * 500.0f; // not used in Cook-Torrance
    _roughness       = roughness;
    _metalness       = metalness;
    _numTextures     = 0;
    _getsShadows     = true;
    _reflectionModel = RM_CookTorrance;
    _skybox          = skybox;
    _program         = program;

    _kr = 0.0f;
    _kt = 0.0f;
    _kn = 1.0f;

    // Add pointer to the global resource vectors for deallocation
    if (am)
        am->materials().push_back(this);
}
//-----------------------------------------------------------------------------
/*! Constructor for Cook-Torrance shaded materials with PBR textures.
 Materials can be used by multiple meshes (SLMesh). Materials can belong
 therefore to the global assets such as meshes, materials, textures and
 shader programs.
 @param am Pointer to a global asset manager. If passed the asset manager
 is the owner of the instance and will do the deallocation. If a nullptr
 is passed the creator is responsible for the deallocation.
 @param name Name of the material
 @param skybox Pointer to the skybox if available. If the skybox is an HDR
 skybox it will influence the ambient and specular reflection.
 @param texture1 Pointer to a SLGLTexture of a specific SLTextureType. For
 PBR materials this can be TT_diffuse, TT_normal, TT_roughness, TT_metallic
 and TT_occlusion.
 @param texture2 Pointer to a SLGLTexture of a specific SLTextureType.
 @param texture3 Pointer to a SLGLTexture of a specific SLTextureType.
 @param texture4 Pointer to a SLGLTexture of a specific SLTextureType.
 @param texture5 Pointer to a SLGLTexture of a specific SLTextureType.
 @param program Pointer to the shader program for the material.
 If none is passed a program will be generated from the passed parameters.
 */
SLMaterial::SLMaterial(SLAssetManager* am,
                       const SLchar*   name,
                       SLSkybox*       skybox,
                       SLGLTexture*    texture1,
                       SLGLTexture*    texture2,
                       SLGLTexture*    texture3,
                       SLGLTexture*    texture4,
                       SLGLTexture*    texture5,
                       SLGLProgram*    program) : SLObject(name)
{
    _assetManager = am;
    _ambient.set(1, 1, 1);
    _diffuse.set(1, 1, 1);
    _specular.set(1, 1, 1);
    _emissive.set(0, 0, 0, 0);
    _shininess       = 125;
    _roughness       = 0.5f;
    _metalness       = 0.0f;
    _numTextures     = 0;
    _reflectionModel = RM_CookTorrance;
    _getsShadows     = true;
    _skybox          = skybox;
    _program         = program;

    addTexture(texture1);
    addTexture(texture2);
    addTexture(texture3);
    addTexture(texture4);
    addTexture(texture5);

    _kr        = 0.0f;
    _kt        = 0.0f;
    _kn        = 1.0f;
    _diffuse.w = 1.0f - _kt;
    _program   = nullptr;

    // Add pointer to the global resource vectors for deallocation
    if (am)
        am->materials().push_back(this);
}
//-----------------------------------------------------------------------------
/*! Constructor for textured particle system materials (Draw=.
 Materials can be used by multiple meshes (SLMesh). Materials can belong
 therefore to the global assets such as meshes, materials, textures and
 shader programs.
 @param am Pointer to a global asset manager. If passed the asset
 manager is the owner of the instance and will do the deallocation. If a
 nullptr is passed the creator is responsible for the deallocation.
 @param name Name of the material
 @param texture Pointer to an SLGLTexture of a specific SLTextureType
 @param ps Pointer to the particle system for the material.
 @param program Pointer to the shader program for the material.
 If none is passed a program will be generated from the passed parameters.
 */
SLMaterial::SLMaterial(SLAssetManager*   am,
                       const SLchar*     name,
                       SLParticleSystem* ps,
                       SLGLTexture*      texture,
                       SLGLProgram*      program,
                       SLGLProgram*      programTF) : SLObject(name)
{
    _assetManager    = am;
    _reflectionModel = RM_Particle;
    _getsShadows     = true; // Later for Particle System maybe
    _skybox          = nullptr;
    _ps              = ps;
    _program         = program;
    _programTF       = programTF;

    _numTextures = 0;
    addTexture(texture);

    // Add pointer to the global resource vectors for deallocation
    if (am)
        am->materials().push_back(this);
}
//-----------------------------------------------------------------------------
/*! Constructor for materials with only a shader program.
 Materials can be used by multiple meshes (SLMesh). Materials can belong
 therefore to the global assets such as meshes, materials, textures and
 shader programs.
 @param am Pointer to a global asset manager. If passed the asset
 manager is the owner of the instance and will do the deallocation. If a
 nullptr is passed the creator is responsible for the deallocation.
 @param name Name of the material
 @param shaderProg Pointer to the shader program for the material
 */
SLMaterial::SLMaterial(SLAssetManager* am,
                       const SLchar*   name,
                       SLGLProgram*    shaderProg) : SLObject(name)
{
    _assetManager    = am;
    _program         = shaderProg;
    _skybox          = nullptr;
    _reflectionModel = RM_BlinnPhong;
    _ambient.set(1, 1, 1);
    _diffuse.set(1, 1, 1);
    _specular.set(1, 1, 1);
    _emissive.set(0, 0, 0, 0);
    _shininess    = 125.0f;
    _roughness    = 0.0f;
    _metalness    = 0.0f;
    _translucency = 0.0f;
    _getsShadows  = true;
    _numTextures  = 0;

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
 */
SLMaterial::SLMaterial(SLAssetManager* am,
                       SLGLProgram*    colorUniformProgram,
                       const SLCol4f&  uniformColor,
                       const SLchar*   name) : SLObject(name)
{
    _assetManager    = am;
    _reflectionModel = RM_Custom;
    _ambient.set(0, 0, 0);
    _diffuse = uniformColor;
    _specular.set(0, 0, 0);
    _emissive.set(0, 0, 0, 0);
    _shininess    = 125;
    _roughness    = 0.5f;
    _metalness    = 0.0f;
    _translucency = 0.0f;
    _program      = colorUniformProgram;
    _skybox       = nullptr;
    _kr           = 0.0f;
    _kt           = 0.0f;
    _kn           = 1.0f;
    _getsShadows  = true;
    _numTextures  = 0;

    // Add pointer to the global resource vectors for deallocation
    if (am)
        am->materials().push_back(this);
}
//-----------------------------------------------------------------------------
//! Adds the passed texture to the equivalent texture type vector
void SLMaterial::addTexture(SLGLTexture* texture)
{
    if (!texture)
        return;

    if (texture->target() == GL_TEXTURE_3D)
        _textures3d.push_back(texture);

    _textures[texture->texType()].push_back(texture);

    _numTextures++;
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
}
//-----------------------------------------------------------------------------
/*!
 If this material has not yet a shader program assigned (SLMaterial::_program)
 a suitable program will be generated with an instance of SLGLProgramGenerated.
 */
void SLMaterial::generateProgramPS()
{
    // If no shader program is attached add a generated shader program
    // A 3D object can be stored without material or shader program information.
    if (!_program)
    {
        /////////////////////////////
        // Generate draw program
        /////////////////////////////

        // Check first the asset manager if the requested program type already exists
        string programNameDraw;
        SLGLProgramGenerated::buildProgramNamePS(this, programNameDraw, true);
        _program = _assetManager->getProgramByName(programNameDraw);

        // If the program was not found by name generate a new one
        if (!_program)
        {
            _program = new SLGLProgramGenerated(_assetManager,
                                                programNameDraw,
                                                this,
                                                true,
                                                "Geom");
        }
    }

    if (!_programTF)
    {
        /////////////////////////////
        // Generate update program
        /////////////////////////////

        // Check first the asset manager if the requested programTF type already exists
        string programNameUpdate;
        SLGLProgramGenerated::buildProgramNamePS(this, programNameUpdate, false);
        _programTF = _assetManager->getProgramByName(programNameUpdate);
        if (!_programTF)
        {
            _programTF = new SLGLProgramGenerated(_assetManager,
                                                  programNameUpdate,
                                                  this,
                                                  false);

            int                 countString = 3;
            vector<const char*> outputNames; // For transform feedback
            outputNames.push_back("tf_position");
            outputNames.push_back("tf_velocity");
            outputNames.push_back("tf_startTime");
            if (_ps->doAcc() || _ps->doGravity())
            {
                outputNames.push_back("tf_initialVelocity");
                countString++;
            }
            if (_ps->doRotation())
            {
                outputNames.push_back("tf_rotation");
                countString++;
            }
            if (_ps->doRotation() && _ps->doRotRange())
            {
                outputNames.push_back("tf_angularVelo");
                countString++;
            }
            if (_ps->doFlipBookTexture())
            {
                outputNames.push_back("tf_texNum");
                countString++;
            }
            if (_ps->doShape())
            {
                outputNames.push_back("tf_initialPosition");
                countString++;
            }
            _programTF->initTF(&outputNames[0], countString);
        }
    }

    // Check if shader had a compile error and the error texture should be shown
    if (_program && _program->name().find("ErrorTex") != string::npos)
    {
        for (int i = 0; i < TT_numTextureType; i++)
            _textures[i].clear();
        if (!_errorTexture && !_compileErrorTexFilePath.empty())
            _errorTexture = new SLGLTexture(nullptr, _compileErrorTexFilePath);
        _textures[TT_diffuse].push_back(_errorTexture);
    }

    if (_programTF && _programTF->name().find("ErrorTex") != string::npos)
    {
        for (int i = 0; i < TT_numTextureType; i++)
            _textures[i].clear();
        if (!_errorTexture && !_compileErrorTexFilePath.empty())
            _errorTexture = new SLGLTexture(nullptr, _compileErrorTexFilePath);
        _textures[TT_diffuse].push_back(_errorTexture);
    }
}
//-----------------------------------------------------------------------------
/*!
 SLMaterial::activate activates this material for rendering if it is not yet
 the active one and set as SLGLState::currentMaterial. If this material has
 not yet a shader program assigned (SLMaterial::_program) a suitable program
 will be generated with an instance of SLGLProgramGenerated.
 At the end the shader program will begin its usage with SLGLProgram::beginUse.
 @param cam Pointer to the active camera
 @param lights Pointer to the scene vector of lights
 @param skybox Pointer to the skybox
 */
void SLMaterial::activate(SLCamera* cam, SLVLight* lights, SLSkybox* skybox)
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

    // Check if shader had a compile error and the error texture should be shown
    if (_program && _program->name().find("ErrorTex") != string::npos)
    {
        for (int i = 0; i < TT_numTextureType; i++)
            _textures[i].clear();
        if (!_errorTexture && !_compileErrorTexFilePath.empty())
            _errorTexture = new SLGLTexture(nullptr, _compileErrorTexFilePath);
        _textures[TT_diffuse].push_back(_errorTexture);
    }

    // Activate the shader program now
    _program->beginUse(cam, this, lights);
}
//-----------------------------------------------------------------------------
//! Passes all material parameters as uniforms to the passed shader program
SLint SLMaterial::passToUniforms(SLGLProgram* program, SLint nextTexUnit)
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
    program->uniform1i("u_matHasTexture", _numTextures > 0 ? 1 : 0);

    // pass textures unit id to the sampler uniform
    for (SLuint i = 0; i < TT_numTextureType; i++)
    {
        int texNb = 0;
        for (SLGLTexture* texture : _textures[i])
        {
            SLchar name[100];
            texture->bindActive(nextTexUnit);
            switch (i)
            {
                case TT_diffuse:
                {
                    snprintf(name, sizeof(name), "u_matTextureDiffuse%d", texNb);
                    break;
                }
                case TT_specular:
                {
                    snprintf(name, sizeof(name), "u_matTextureSpecular%d", texNb);
                    break;
                }
                case TT_normal:
                {
                    snprintf(name, sizeof(name), "u_matTextureNormal%d", texNb);
                    break;
                }
                case TT_height:
                {
                    snprintf(name, sizeof(name), "u_matTextureHeight%d", texNb);
                    break;
                }
                case TT_occlusion:
                {
                    snprintf(name, sizeof(name), "u_matTextureOcclusion%d", texNb);
                    break;
                }
                case TT_roughness:
                {
                    snprintf(name, sizeof(name), "u_matTextureRoughness%d", texNb);
                    break;
                }
                case TT_metallic:
                {
                    snprintf(name, sizeof(name), "u_matTextureMetallic%d", texNb);
                    break;
                }
                case TT_roughMetal:
                {
                    snprintf(name, sizeof(name), "u_matTextureRoughMetal%d", texNb);
                    break;
                }
                case TT_occluRoughMetal:
                {
                    snprintf(name, sizeof(name), "u_matTextureOccluRoughMetal%d", texNb);
                    break;
                }
                case TT_emissive:
                {
                    snprintf(name, sizeof(name), "u_matTextureEmissive%d", texNb);
                    break;
                }
                case TT_environmentCubemap:
                {
                    snprintf(name, sizeof(name), "u_matTextureEnvCubemap%d", texNb);
                    break;
                }
                case TT_font:
                {
                    snprintf(name, sizeof(name), "u_matTextureFont%d", texNb);
                    break;
                }
                default:
                {
                    snprintf(name, sizeof(name), "u_matTextureDiffuse%d", texNb);
                    break;
                }
            }

            if (program->uniform1i(name, nextTexUnit) < 0)
                Utils::log("Material", "texture name %s not found for program: %s", name, program->name().c_str());

            texNb++;
            nextTexUnit++;
        }
    }

    // Pass environment mapping uniforms from the skybox
    if (_skybox &&
        _skybox->irradianceCubemap() &&
        _skybox->roughnessCubemap() &&
        _skybox->brdfLutTexture())
    {
        if (program->uniform1i("u_skyIrradianceCubemap", nextTexUnit) >= 0)
            _skybox->irradianceCubemap()->bindActive(nextTexUnit++);

        if (program->uniform1i("u_skyRoughnessCubemap", nextTexUnit) >= 0)
            _skybox->roughnessCubemap()->bindActive(nextTexUnit++);

        if (program->uniform1i("u_skyBrdfLutTexture", nextTexUnit) >= 0)
            _skybox->brdfLutTexture()->bindActive(nextTexUnit++);

        program->uniform1f("u_skyExposure", _skybox->exposure());
    }

    return nextTexUnit;
}
//-----------------------------------------------------------------------------
//! Returns a unique string that represent all textures used
SLstring SLMaterial::texturesString()
{
    SLstring texStr;
    for (SLuint iTT = 0; iTT < TT_numTextureType; ++iTT)
    {
        for (SLuint iT = 0; iT < _textures[iTT].size(); ++iT)
        {
            texStr += "-" +
                      _textures[iTT][iT]->typeShortName() +
                      std::to_string(iT) +
                      std::to_string(_textures[iTT][iT]->uvIndex());
        }
    }
    if (_skybox)
        texStr += "-Sky";

    return texStr;
}
//-----------------------------------------------------------------------------
//! Returns true if the specified uvIndex is used by one of the textures
SLbool SLMaterial::usesUVIndex(SLbyte uvIndex)
{
    for (SLuint iTT = 0; iTT < TT_numTextureType; ++iTT)
    {
        for (SLuint iT = 0; iT < _textures[iTT].size(); ++iT)
        {
            if (_textures[iTT][iT]->uvIndex() == uvIndex)
                return true;
        }
    }
    return false;
}
//-----------------------------------------------------------------------------
