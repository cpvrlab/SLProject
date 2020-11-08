//#############################################################################
//  File:      SLGLProgram.cpp
//  Author:    Marcus Hudritsch
//             Mainly based on Martin Christens GLSL Tutorial
//             See http://www.clockworkcoders.com
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <SLAssetManager.h>
#include <SLGLDepthBuffer.h>
#include <SLGLProgram.h>
#include <SLGLShader.h>
#include <SLGLState.h>
#include <SLScene.h>
#include <SLAssetManager.h>
#include <SLGLProgramManager.h>
#include <SLGLDepthBuffer.h>

//-----------------------------------------------------------------------------
// Error Strings defined in SLGLShader.h
extern char* aGLSLErrorString[];
//-----------------------------------------------------------------------------
//! Ctor with a vertex and a fragment shader filename.
/*!
 * Constructor for shader programs. Shader programs can be used in multiple
 * materials and can belong therefore to the global assets such as meshes
 * (SLMesh), materials (SLMaterial), textures (SLGLTexture) and shader programs
 * (SLGLProgram).
 * @param s Pointer to a global asset manager. If passed the asset manager is
 * the owner of the instance and will do the deallocation. If a nullptr is passed
 * the creator is responsible for the deallocation.
 * @param vertShaderFile Name of the vertex shader file. If only a filename is
 * passed it will be search on the SLGLProgram::defaultPath.
 * @param fragShaderFile Name of the fragment shader file. If only a filename is
 * passed it will be search on the SLGLProgram::defaultPath.
 * @param geomShaderFile Name of the geometry shader file. If only a filename is
 * passed it will be search on the SLGLProgram::defaultPath.
 */
SLGLProgram::SLGLProgram(SLAssetManager* s,
                         const SLstring& vertShaderFile,
                         const SLstring& fragShaderFile,
                         const SLstring& geomShaderFile) : SLObject("")
{
    _isLinked = false;
    _progID   = 0;

    // optional load vertex and/or fragment shaders
    addShader(new SLGLShader(vertShaderFile, ST_vertex));
    addShader(new SLGLShader(fragShaderFile, ST_fragment));

    if (!geomShaderFile.empty())
        addShader(new SLGLShader(geomShaderFile, ST_geometry));

    // Add pointer to the global resource vectors for deallocation
    if (s)
        s->programs().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
 * The destructor should be called by the owner of the shader program. If an
 * asset manager was passed in the constructor it will do it after scene destruction.
 * The destructor deletes all shader objects (SLGLShader) in the RAM as well
 * as on the GPU.
*/
SLGLProgram::~SLGLProgram()
{
    //SL_LOG("~SLGLProgram");

    for (auto shader : _shaders)
    {
        if (_isLinked)
        {
            glDetachShader(_progID, shader->_shaderID);
            GET_GL_ERROR;
        }

        // always delete shader objects before program object
        delete shader;
    }

    if (_progID > 0)
    {
        glDeleteProgram(_progID);
        GET_GL_ERROR;
    }

    // delete uniform variables
    for (auto uf : _uniforms1f)
        delete uf;
    for (auto ui : _uniforms1i)
        delete ui;
}
//-----------------------------------------------------------------------------
//! SLGLProgram::addShader adds a shader to the shader list
void SLGLProgram::addShader(SLGLShader* shader)
{
    assert(shader);
    _shaders.push_back(shader);
}
//-----------------------------------------------------------------------------
/*! SLGLProgram::initRaw() does not replace any code from the shader and
assumes valid syntax for the shader used. Used in SLGLConetracer
*/
void SLGLProgram::initRaw()
{
    // create program object if it doesn't exist
    if (!_progID)
        _progID = glCreateProgram();

    for (auto* shader : _shaders)
        shader->createAndCompileSimple();

    for (auto* shader : _shaders)
        glAttachShader(_progID, shader->_shaderID);

    GET_GL_ERROR;

    glLinkProgram(_progID);

    GLint success = 0;
    glGetProgramiv(_progID, GL_LINK_STATUS, &success);

    if (!success)
    {
        GLchar log[1024];
        glGetProgramInfoLog(_progID, 1024, nullptr, log);
        std::cerr << "- Failed to link program (" << _progID << ")." << std::endl;
        std::cerr << "LOG: " << std::endl
                  << log << std::endl;
    }

    for (auto* shader : _shaders)
    {
        glDeleteShader(shader->_shaderID);
        GET_GL_ERROR;
    }
}
//-----------------------------------------------------------------------------
/*! SLGLProgram::init creates the OpenGL shaderprogram object, compiles all
shader objects and attaches them to the shaderprogram. At the end all shaders
are linked. If a shader fails to compile a simple texture only shader is
compiled that shows an error message in the texture.
*/
void SLGLProgram::init(SLVLight* lights)
{
    // create program object if it doesn't exist
    if (!_progID)
        _progID = glCreateProgram();

    // if already linked, detach, recreate and compile shaders
    if (_isLinked)
    {
        for (auto* shader : _shaders)
        {
            if (_isLinked)
            {
                glDetachShader(_progID, shader->_shaderID);
                GET_GL_ERROR;
            }
        }
        _isLinked = false;
    }

    // compile all shader objects
    SLbool allSuccuessfullyCompiled = true;
    for (auto* shader : _shaders)
    {
        if (!shader->createAndCompile(lights))
        {
            allSuccuessfullyCompiled = false;
            break;
        }
        GET_GL_ERROR;
    }

    // try to compile alternative per vertex lighting shaders
    if (!allSuccuessfullyCompiled)
    {
        // delete all shaders and uniforms that where attached
        for (auto* sh : _shaders)
            delete sh;
        for (auto* uf : _uniforms1f)
            delete uf;
        for (auto* ui : _uniforms1i)
            delete ui;

        _shaders.clear();
        _uniforms1f.clear();
        _uniforms1i.clear();
    }

    // attach all shader objects
    if (allSuccuessfullyCompiled)
    {
        for (auto* shader : _shaders)
        {
            glAttachShader(_progID, shader->_shaderID);
            GET_GL_ERROR;
        }
    }
    else
        SL_EXIT_MSG("No successfully compiled shaders attached!");

    int linked = 0;
    glLinkProgram(_progID);
    GET_GL_ERROR;
    glGetProgramiv(_progID, GL_LINK_STATUS, &linked);
    GET_GL_ERROR;

    if (linked)
    {
        _isLinked = true;
        for (auto* shader : _shaders)
            _name += shader->name() + ", ";
    }
    else
    {
        SLchar log[256];
        glGetProgramInfoLog(_progID, sizeof(log), nullptr, &log[0]);
        SL_LOG("*** LINKER ERROR ***");
        SL_LOG("Source files: ");
        for (auto* shader : _shaders)
            SL_LOG("%s", shader->name().c_str());
        SL_LOG("%s", log);
        SL_EXIT_MSG("GLSL linker error");
    }
}
//-----------------------------------------------------------------------------
/*! SLGLProgram::useProgram inits the first time the program and then uses it.
Call this initialization if you pass your own custom uniform variables.
*/
void SLGLProgram::useProgram()
{
    if (_progID == 0 && !_shaders.empty())
        init(nullptr);

    if (_isLinked)
    {
        SLGLState::instance()->useProgram(_progID);
        GET_GL_ERROR;
    }
}
//-----------------------------------------------------------------------------
/*! SLGLProgram::beginUse starts using the shader program and transfers the
the camera,  lights and material parameter as uniform variables. It also passes
the custom uniform variables of the _uniform1fList as well as the texture names.
*/
void SLGLProgram::beginUse(SLCamera* cam, SLMaterial* mat, SLVLight* lights)
{
    if (_progID == 0 && !_shaders.empty())
        init(lights);

    if (_isLinked)
    {
        SLGLState* stateGL = SLGLState::instance();

        stateGL->useProgram(_progID);

        if (lights)
            passLightsToUniforms(lights, mat->textures().size());

        if (mat)
            mat->passToUniforms(this);

        if (cam)
            cam->passToUniforms(this);

        for (auto* uf : _uniforms1f)
            uniform1f(uf->name(), uf->value());

        for (auto* ui : _uniforms1i)
            uniform1i(ui->name(), ui->value());

        GET_GL_ERROR;
    }
}
//-----------------------------------------------------------------------------
void SLGLProgram::passLightsToUniforms(SLVLight* lights,
                                       SLuint    numTexInMat) const
{
    SLGLState* stateGL = SLGLState::instance();

    // Pass global lighting value
    uniform1f("u_oneOverGamma", SLLight::oneOverGamma());
    uniform4fv("u_globalAmbi", 1, (const SLfloat*)&SLLight::globalAmbient);

    if (!lights->empty())
    {
        SLMat4f viewRotMat(stateGL->viewMatrix);
        viewRotMat.translation(0, 0, 0); // delete translation part, only rotation needed

        // lighting parameter arrays
        SLint            lightIsOn[SL_MAX_LIGHTS];              //!< flag if light is on
        SLVec4f          lightPosWS[SL_MAX_LIGHTS];             //!< position of light in world space
        SLVec4f          lightPosVS[SL_MAX_LIGHTS];             //!< position of light in view space
        SLVec4f          lightAmbient[SL_MAX_LIGHTS];           //!< ambient light intensity (Ia)
        SLVec4f          lightDiffuse[SL_MAX_LIGHTS];           //!< diffuse light intensity (Id)
        SLVec4f          lightSpecular[SL_MAX_LIGHTS];          //!< specular light intensity (Is)
        SLVec3f          lightSpotDirWS[SL_MAX_LIGHTS];         //!< spot direction in world space
        SLVec3f          lightSpotDirVS[SL_MAX_LIGHTS];         //!< spot direction in view space
        SLfloat          lightSpotCutoff[SL_MAX_LIGHTS];        //!< spot cutoff angle 1-180 degrees
        SLfloat          lightSpotCosCut[SL_MAX_LIGHTS];        //!< cosine of spot cutoff angle
        SLfloat          lightSpotExp[SL_MAX_LIGHTS];           //!< spot exponent
        SLVec3f          lightAtt[SL_MAX_LIGHTS];               //!< att. factor (const,linear,quadratic)
        SLint            lightDoAtt[SL_MAX_LIGHTS];             //!< flag if att. must be calculated
        SLint            lightCreatesShadows[SL_MAX_LIGHTS];    //!< flag if light creates shadows
        SLint            lightDoSmoothShadows[SL_MAX_LIGHTS];   //!< flag if percentage-closer filtering is enabled
        SLuint           lightSmoothShadowLevel[SL_MAX_LIGHTS]; //!< radius of area to sample
        SLfloat          lightShadowMinBias[SL_MAX_LIGHTS];     //!< shadow mapping min. bias at 0 deg.
        SLfloat          lightShadowMaxBias[SL_MAX_LIGHTS];     //!< shadow mapping max. bias at 90 deg.
        SLint            lightUsesCubemap[SL_MAX_LIGHTS];       //!< flag if light has a cube shadow map
        SLMat4f          lightSpace[SL_MAX_LIGHTS * 6];         //!< projection matrix of the light
        SLGLDepthBuffer* lightShadowMap[SL_MAX_LIGHTS];         //!< pointers to depth-buffers for shadow mapping

        // On MacOS and Android the shader for shadow mapping does not work unless
        // all the cube-maps are set. So we define 8 dummy shadow maps for the unused
        // cubemap-SM when the singlemap-SM is used and vice versa.
        static SLGLDepthBuffer unusedSMBuffers[] = {
          SLGLDepthBuffer(SLVec2i(1, 1)),
          SLGLDepthBuffer(SLVec2i(1, 1)),
          SLGLDepthBuffer(SLVec2i(1, 1)),
          SLGLDepthBuffer(SLVec2i(1, 1)),
          SLGLDepthBuffer(SLVec2i(1, 1)),
          SLGLDepthBuffer(SLVec2i(1, 1)),
          SLGLDepthBuffer(SLVec2i(1, 1)),
          SLGLDepthBuffer(SLVec2i(1, 1)),
        };

        // Init to defaults
        for (SLint i = 0; i < SL_MAX_LIGHTS; ++i)
        {
            lightIsOn[i]       = 0;
            lightPosWS[i]      = SLVec4f(0, 0, 1, 1);
            lightPosVS[i]      = SLVec4f(0, 0, 1, 1);
            lightAmbient[i]    = SLCol4f::BLACK;
            lightDiffuse[i]    = SLCol4f::BLACK;
            lightSpecular[i]   = SLCol4f::BLACK;
            lightSpotDirWS[i]  = SLVec3f(0, 0, -1);
            lightSpotDirVS[i]  = SLVec3f(0, 0, -1);
            lightSpotCutoff[i] = 180.0f;
            lightSpotCosCut[i] = cos(Utils::DEG2RAD * lightSpotCutoff[i]);
            lightSpotExp[i]    = 1.0f;
            lightAtt[i].set(1.0f, 0.0f, 0.0f);
            lightDoAtt[i] = 0;
            for (SLint ii = 0; ii < 6; ++ii)
                lightSpace[i * 6 + ii] = SLMat4f();
            lightCreatesShadows[i]    = 0;
            lightDoSmoothShadows[i]   = 0;
            lightSmoothShadowLevel[i] = 1;
            lightShadowMinBias[i]     = 0.001f;
            lightShadowMaxBias[i]     = 0.008f;
            lightUsesCubemap[i]       = 0;
            lightShadowMap[i]         = nullptr;
        }

        // Fill up light property vectors
        for (SLuint i = 0; i < lights->size(); ++i)
        {
            SLLight*     light     = lights->at(i);
            SLShadowMap* shadowMap = light->shadowMap();

            lightIsOn[i]  = light->isOn();
            SLVec4f posWS = light->positionWS();
            lightPosWS[i].set(posWS);
            SLVec4f posVS = stateGL->viewMatrix * lightPosWS[i];
            lightPosVS[i].set(posVS);
            lightAmbient[i].set(light->ambient());
            lightDiffuse[i].set(light->diffuse());
            lightSpecular[i].set(light->specular());
            SLVec3f dirWS = light->spotDirWS();
            lightSpotDirWS[i].set(dirWS);
            SLVec3f dirVS = viewRotMat.multVec(lightSpotDirWS[i]);
            lightSpotDirVS[i].set(dirVS);
            lightSpotCutoff[i]        = light->spotCutOffDEG();
            lightSpotCosCut[i]        = light->spotCosCut();
            lightSpotExp[i]           = light->spotExponent();
            lightAtt[i]               = SLVec3f(light->kc(), light->kl(), light->kq());
            lightDoAtt[i]             = light->isAttenuated();
            lightCreatesShadows[i]    = light->createsShadows();
            lightDoSmoothShadows[i]   = light->doSoftShadows();
            lightSmoothShadowLevel[i] = light->softShadowLevel();
            lightShadowMinBias[i]     = light->shadowMinBias();
            lightShadowMaxBias[i]     = light->shadowMaxBias();
            lightUsesCubemap[i]       = shadowMap && shadowMap->useCubemap() ? 1 : 0;
            lightShadowMap[i]         = shadowMap && shadowMap->depthBuffer() ? shadowMap->depthBuffer() : nullptr;
            if (lightShadowMap[i])
                for (SLint ls = 0; ls < 6; ++ls)
                    lightSpace[i * 6 + ls] = shadowMap->mvp()[ls];
        }

        // Pass vectors as uniform vectors
        auto nL = (SLint)lights->size();
        uniform1iv("u_lightIsOn", nL, (SLint*)&lightIsOn);
        uniform4fv("u_lightPosWS", nL, (SLfloat*)&lightPosWS);
        uniform4fv("u_lightPosVS", nL, (SLfloat*)&lightPosVS);
        uniform4fv("u_lightAmbi", nL, (SLfloat*)&lightAmbient);
        uniform4fv("u_lightDiff", nL, (SLfloat*)&lightDiffuse);
        uniform4fv("u_lightSpec", nL, (SLfloat*)&lightSpecular);
        uniform3fv("u_lightSpotDir", nL, (SLfloat*)&lightSpotDirVS);
        uniform1fv("u_lightSpotDeg", nL, (SLfloat*)&lightSpotCutoff);
        uniform1fv("u_lightSpotCos", nL, (SLfloat*)&lightSpotCosCut);
        uniform1fv("u_lightSpotExp", nL, (SLfloat*)&lightSpotExp);
        uniform3fv("u_lightAtt", nL, (SLfloat*)&lightAtt);
        uniform1iv("u_lightDoAtt", nL, (SLint*)&lightDoAtt);
        uniform1iv("u_lightDoSmoothShadows", nL, (SLint*)&lightDoSmoothShadows);
        uniform1iv("u_lightSmoothShadowLevel", nL, (SLint*)&lightSmoothShadowLevel);
        uniform1iv("u_lightUsesCubemap", nL, (SLint*)&lightUsesCubemap);
        uniformMatrix4fv("u_lightSpace", nL * 6, (SLfloat*)&lightSpace);
        uniform1iv("u_lightCreatesShadows", nL, (SLint*)&lightCreatesShadows);
        uniform1fv("u_lightShadowMinBias", nL, (SLfloat*)&lightShadowMinBias);
        uniform1fv("u_lightShadowMaxBias", nL, (SLfloat*)&lightShadowMaxBias);

        for (int i = 0; i < SL_MAX_LIGHTS; ++i)
        {
            if (lightCreatesShadows[i])
            {
                SLint    loc = 0;
                SLstring uniformSM_Used;
                SLstring uniformSM_Unused;
                SLuint   texUnit_Used;
                SLuint   texUnit_Unused;

                if (lightUsesCubemap[i])
                {
                    texUnit_Used     = numTexInMat + SL_MAX_LIGHTS + i;
                    texUnit_Unused   = numTexInMat + i;
                    uniformSM_Used   = "u_shadowMapCube_" + std::to_string(SL_MAX_LIGHTS + i);
                    uniformSM_Unused = "u_shadowMap_" + std::to_string(i);
                }
                else
                {
                    texUnit_Used     = numTexInMat + i;
                    texUnit_Unused   = numTexInMat + SL_MAX_LIGHTS + i;
                    uniformSM_Used   = "u_shadowMap_" + std::to_string(i);
                    uniformSM_Unused = "u_shadowMapCube_" + std::to_string(SL_MAX_LIGHTS + i);
                }

                // Set used SM
                if ((loc = getUniformLocation(uniformSM_Used.c_str())) >= 0)
                {
                    lightShadowMap[i]->bindActive(texUnit_Used);
                    glUniform1i(loc, texUnit_Used);
                }

                // Set unused SM
                if ((loc = getUniformLocation(uniformSM_Unused.c_str())) >= 0)
                {
                    unusedSMBuffers[i].bindActive(texUnit_Unused);
                    glUniform1i(loc, texUnit_Unused);
                }
            }
        }
    }
}
//-----------------------------------------------------------------------------
//! SLGLProgram::endUse stops the shader program
void SLGLProgram::endUse()
{
    SLGLState* stateGL = SLGLState::instance();

    // In core profile you must have an active program
    if (stateGL->glVersionNOf() > 3.0f) return;

    stateGL->useProgram(0);
}
//-----------------------------------------------------------------------------
//! SLGLProgram::addUniform1f add a uniform variable to the list
void SLGLProgram::addUniform1f(SLGLUniform1f* u)
{
    _uniforms1f.push_back(u);
}
//-----------------------------------------------------------------------------
//! SLGLProgram::addUniform1f add a uniform variable to the list
void SLGLProgram::addUniform1i(SLGLUniform1i* u)
{
    _uniforms1i.push_back(u);
}
//-----------------------------------------------------------------------------
SLint SLGLProgram::getUniformLocation(const SLchar* name) const
{
    SLint loc = glGetUniformLocation(_progID, name);
    GET_GL_ERROR;
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the float value v0 to the uniform variable "name"
SLint SLGLProgram::uniform1f(const SLchar* name, SLfloat v0) const
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform1f(loc, v0);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the float values v0 & v1 to the uniform variable "name"
SLint SLGLProgram::uniform2f(const SLchar* name,
                             SLfloat       v0,
                             SLfloat       v1) const
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform2f(loc, v0, v1);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the float values v0, v1 & v2 to the uniform variable "name"
SLint SLGLProgram::uniform3f(const SLchar* name,
                             SLfloat       v0,
                             SLfloat       v1,
                             SLfloat       v2) const
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform3f(loc, v0, v1, v2);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the float values v0,v1,v2 & v3 to the uniform variable "name"
SLint SLGLProgram::uniform4f(const SLchar* name,
                             SLfloat       v0,
                             SLfloat       v1,
                             SLfloat       v2,
                             SLfloat       v3) const
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform4f(loc, v0, v1, v2, v3);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the int values v0 to the uniform variable "name"
SLint SLGLProgram::uniform1i(const SLchar* name, SLint v0) const
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform1i(loc, v0);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the int values v0 & v1 to the uniform variable "name"
SLint SLGLProgram::uniform2i(const SLchar* name,
                             SLint         v0,
                             SLint         v1) const
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform2i(loc, v0, v1);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the int values v0, v1 & v2 to the uniform variable "name"
SLint SLGLProgram::uniform3i(const SLchar* name,
                             SLint         v0,
                             SLint         v1,
                             SLint         v2) const
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform3i(loc, v0, v1, v2);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the int values v0, v1, v2 & v3 to the uniform variable "name"
SLint SLGLProgram::uniform4i(const SLchar* name,
                             SLint         v0,
                             SLint         v1,
                             SLint         v2,
                             SLint         v3) const
{
    SLint loc = getUniformLocation(name);
    if (loc == -1) return false;
    glUniform4i(loc, v0, v1, v2, v3);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 1 float value py pointer to the uniform variable "name"
SLint SLGLProgram::uniform1fv(const SLchar*  name,
                              SLsizei        count,
                              const SLfloat* value) const
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform1fv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 2 float values py pointer to the uniform variable "name"
SLint SLGLProgram::uniform2fv(const SLchar*  name,
                              SLsizei        count,
                              const SLfloat* value) const
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform2fv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 3 float values py pointer to the uniform variable "name"
SLint SLGLProgram::uniform3fv(const SLchar*  name,
                              SLsizei        count,
                              const SLfloat* value) const
{
    SLint loc = getUniformLocation(name);
    if (loc == -1) return false;
    glUniform3fv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 4 float values py pointer to the uniform variable "name"
SLint SLGLProgram::uniform4fv(const SLchar*  name,
                              SLsizei        count,
                              const SLfloat* value) const
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform4fv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 1 int value py pointer to the uniform variable "name"
SLint SLGLProgram::uniform1iv(const SLchar* name,
                              SLsizei       count,
                              const SLint*  value) const
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform1iv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 2 int values py pointer to the uniform variable "name"
SLint SLGLProgram::uniform2iv(const SLchar* name,
                              SLsizei       count,
                              const SLint*  value) const
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform2iv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 3 int values py pointer to the uniform variable "name"
SLint SLGLProgram::uniform3iv(const SLchar* name,
                              SLsizei       count,
                              const SLint*  value) const
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform3iv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 4 int values py pointer to the uniform variable "name"
SLint SLGLProgram::uniform4iv(const SLchar* name,
                              SLsizei       count,
                              const SLint*  value) const
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform4iv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes a 2x2 float matrix values py pointer to the uniform variable "name"
SLint SLGLProgram::uniformMatrix2fv(const SLchar*  name,
                                    SLsizei        count,
                                    const SLfloat* value,
                                    GLboolean      transpose) const
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniformMatrix2fv(loc, count, transpose, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes a 2x2 float matrix values py pointer to the uniform at location loc
void SLGLProgram::uniformMatrix2fv(const SLint    loc,
                                   SLsizei        count,
                                   const SLfloat* value,
                                   GLboolean      transpose) const
{
    glUniformMatrix2fv(loc, count, transpose, value);
}
//-----------------------------------------------------------------------------
//! Passes a 3x3 float matrix values py pointer to the uniform variable "name"
SLint SLGLProgram::uniformMatrix3fv(const SLchar*  name,
                                    SLsizei        count,
                                    const SLfloat* value,
                                    GLboolean      transpose) const
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniformMatrix3fv(loc, count, transpose, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes a 3x3 float matrix values py pointer to the uniform at location loc
void SLGLProgram::uniformMatrix3fv(const SLint    loc,
                                   SLsizei        count,
                                   const SLfloat* value,
                                   GLboolean      transpose) const
{
    glUniformMatrix3fv(loc, count, transpose, value);
}
//-----------------------------------------------------------------------------
//! Passes a 4x4 float matrix values py pointer to the uniform variable "name"
SLint SLGLProgram::uniformMatrix4fv(const SLchar*  name,
                                    SLsizei        count,
                                    const SLfloat* value,
                                    GLboolean      transpose) const
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniformMatrix4fv(loc, count, transpose, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes a 4x4 float matrix values py pointer to the uniform at location loc
void SLGLProgram::uniformMatrix4fv(const SLint    loc,
                                   SLsizei        count,
                                   const SLfloat* value,
                                   GLboolean      transpose) const
{
    glUniformMatrix4fv(loc, count, transpose, value);
}
//-----------------------------------------------------------------------------
