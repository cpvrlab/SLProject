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

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <SLApplication.h>
#include <SLGLProgram.h>
#include <SLGLShader.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
//! Default path for shader files used when only filename is passed in load.
//! Is overwritten in slCreateAppAndScene.
SLstring SLGLProgram::defaultPath = SLstring(SL_PROJECT_ROOT) + "/data/shaders";
//-----------------------------------------------------------------------------
// Error Strings defined in SLGLShader.h
extern char* aGLSLErrorString[];
//-----------------------------------------------------------------------------
//! Ctor with a vertex and a fragment shader filename.
SLGLProgram::SLGLProgram(SLstring vertShaderFile,
                         SLstring fragShaderFile,
                         SLstring geomShaderFile) : SLObject("")
{
    _isLinked = false;
    _progID   = 0;

    // optional load vertex and/or fragment shaders
    addShader(new SLGLShader(defaultPath + vertShaderFile, ST_vertex));
    addShader(new SLGLShader(defaultPath + fragShaderFile, ST_fragment));
    if (!geomShaderFile.empty())
        addShader(new SLGLShader(defaultPath + geomShaderFile, ST_geometry));

    // Add pointer to the global resource vectors for deallocation
    SLApplication::scene->programs().push_back(this);
}
//-----------------------------------------------------------------------------
//! The destructor detaches all shader objects and deletes them
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

    for (auto shader : _shaders)
        shader->createAndCompileSimple();

    for (auto shader : _shaders)
        glAttachShader(_progID, shader->_shaderID);

    GET_GL_ERROR;

    glLinkProgram(_progID);

    GLint success;
    glGetProgramiv(_progID, GL_LINK_STATUS, &success);

    if (!success)
    {
        GLchar log[1024];
        glGetProgramInfoLog(_progID, 1024, nullptr, log);
        std::cerr << "- Failed to link program (" << _progID << ")." << std::endl;
        std::cerr << "LOG: " << std::endl
                  << log << std::endl;
    }

    for (auto shader : _shaders)
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
void SLGLProgram::init()
{
    // create program object if it doesn't exist
    if (!_progID) _progID = glCreateProgram();

    // if already linked, detach, recreate and compile shaders
    if (_isLinked)
    {
        for (auto shader : _shaders)
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
    for (auto shader : _shaders)
    {
        if (!shader->createAndCompile())
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
        for (auto sh : _shaders)
            delete sh;
        for (auto uf : _uniforms1f)
            delete uf;
        for (auto ui : _uniforms1i)
            delete ui;
        _shaders.clear();
        _uniforms1f.clear();
        _uniforms1i.clear();

        addShader(new SLGLShader(defaultPath + "ErrorTex.vert", ST_vertex));
        addShader(new SLGLShader(defaultPath + "ErrorTex.frag", ST_fragment));

        allSuccuessfullyCompiled = true;
        for (auto shader : _shaders)
        {
            if (!shader->createAndCompile())
            {
                allSuccuessfullyCompiled = false;
                break;
            }
            GET_GL_ERROR;
        }
    }

    // attach all shader objects
    if (allSuccuessfullyCompiled)
    {
        for (auto shader : _shaders)
        {
            glAttachShader(_progID, shader->_shaderID);
            GET_GL_ERROR;
        }
    }
    else
        SL_EXIT_MSG("No successufully compiled shaders attached!");

    int linked;
    glLinkProgram(_progID);
    GET_GL_ERROR;
    glGetProgramiv(_progID, GL_LINK_STATUS, &linked);
    GET_GL_ERROR;

    if (linked)
    {
        _isLinked = true;
        for (auto shader : _shaders)
            _name += "+" + shader->name();
        //SL_LOG("Linked: %s", _name.c_str());
    }
    else
    {
        SLchar log[256];
        glGetProgramInfoLog(_progID, sizeof(log), nullptr, &log[0]);
        SL_LOG("*** LINKER ERROR ***");
        SL_LOG("Source files: ");
        for (auto shader : _shaders)
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
    if (_progID == 0 && !_shaders.empty()) init();

    if (_isLinked)
    {
        SLGLState::instance()->useProgram(_progID);
        GET_GL_ERROR;
    }
}
//-----------------------------------------------------------------------------
/*! SLGLProgram::beginUse starts using the shaderprogram and transfers the
the standard light and material parameter as uniform variables. It also passes 
the custom uniform variables of the _uniform1fList as well as the texture names.
*/
void SLGLProgram::beginUse(SLMaterial* mat)
{
    assert(mat != nullptr && "SLGLProgram::beginUse: No material passed.");

    if (_progID == 0 && !_shaders.empty()) init();

    if (_isLinked)
    {
        SLGLState* stateGL = SLGLState::instance();

        // 1: Activate the shader program object
        stateGL->useProgram(_progID);

        // 2: Pass light & material parameters
        stateGL->globalAmbientLight = SLApplication::scene->globalAmbiLight();
        SLint loc                   = uniform4fv("u_globalAmbient", 1, (const SLfloat*)stateGL->globalAmbient());
        loc                         = uniform1i("u_numLightsUsed", stateGL->numLightsUsed);

        if (stateGL->numLightsUsed > 0)
        {
            SLint nL = SL_MAX_LIGHTS;
            stateGL->calcLightPosVS(stateGL->numLightsUsed);
            stateGL->calcLightDirVS(stateGL->numLightsUsed);
            loc = uniform1iv("u_lightIsOn", nL, (SLint*)stateGL->lightIsOn);
            loc = uniform4fv("u_lightPosVS", nL, (SLfloat*)stateGL->lightPosVS);
            loc = uniform4fv("u_lightAmbient", nL, (SLfloat*)stateGL->lightAmbient);
            loc = uniform4fv("u_lightDiffuse", nL, (SLfloat*)stateGL->lightDiffuse);
            loc = uniform4fv("u_lightSpecular", nL, (SLfloat*)stateGL->lightSpecular);
            loc = uniform3fv("u_lightSpotDirVS", nL, (SLfloat*)stateGL->lightSpotDirVS);
            loc = uniform1fv("u_lightSpotCutoff", nL, (SLfloat*)stateGL->lightSpotCutoff);
            loc = uniform1fv("u_lightSpotCosCut", nL, (SLfloat*)stateGL->lightSpotCosCut);
            loc = uniform1fv("u_lightSpotExp", nL, (SLfloat*)stateGL->lightSpotExp);
            loc = uniform3fv("u_lightAtt", nL, (SLfloat*)stateGL->lightAtt);
            loc = uniform1iv("u_lightDoAtt", nL, (SLint*)stateGL->lightDoAtt);

            mat->passToUniforms(this);
        }

        // 2b: Set stereo states
        loc = uniform1i("u_projection", stateGL->projection);
        loc = uniform1i("u_stereoEye", stateGL->stereoEye);
        loc = uniformMatrix3fv("u_stereoColorFilter", 1, (SLfloat*)&stateGL->stereoColorFilter);

        // 2c: Pass diffuse color for uniform color shader
        SLCol4f diffuse = mat->diffuse();
        loc             = uniform4fv("u_color", 1, (SLfloat*)&diffuse);

        // 2d: Pass gamma correction value
        loc = uniform1f("u_oneOverGamma", stateGL->oneOverGamma);

        // 3: Pass the custom uniform1f variables of the list
        for (auto uf : _uniforms1f)
            loc = uniform1f(uf->name(), uf->value());
        for (auto ui : _uniforms1i)
            loc = uniform1i(ui->name(), ui->value());

        // 4: Send texture units as uniforms texture samplers
        for (SLint i = 0; i < (SLint)mat->textures().size(); ++i)
        {
            SLchar name[100];
            sprintf(name, "u_texture%d", i);
            loc = uniform1i(name, i);
        }
        GET_GL_ERROR;
    }
}
//-----------------------------------------------------------------------------
//! SLGLProgram::endUse stops the shaderprogram
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
SLint SLGLProgram::getUniformLocation(const SLchar* name)
{
    SLint loc = glGetUniformLocation(_progID, name);
#ifdef _GLDEBUG
    GET_GL_ERROR;
#endif
    return loc;
}
//-----------------------------------------------------------------------------
SLint SLGLProgram::getAttribLocation(const SLchar* name)
{
    SLint loc = glGetAttribLocation(_progID, name);
#ifdef _GLDEBUG
    GET_GL_ERROR;
#endif
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the float value v0 to the uniform variable "name"
SLint SLGLProgram::uniform1f(const SLchar* name, SLfloat v0)
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform1f(loc, v0);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the float values v0 & v1 to the uniform variable "name"
SLint SLGLProgram::uniform2f(const SLchar* name, SLfloat v0, SLfloat v1)
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
                             SLfloat       v2)
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
                             SLfloat       v3)
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform4f(loc, v0, v1, v2, v3);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the int values v0 to the uniform variable "name"
SLint SLGLProgram::uniform1i(const SLchar* name, SLint v0)
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform1i(loc, v0);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the int values v0 & v1 to the uniform variable "name"
SLint SLGLProgram::uniform2i(const SLchar* name,
                             SLint         v0,
                             SLint         v1)
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
                             SLint         v2)
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
                             SLint         v3)
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
                              const SLfloat* value)
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform1fv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 2 float values py pointer to the uniform variable "name"
SLint SLGLProgram::uniform2fv(const SLchar*  name,
                              SLsizei        count,
                              const SLfloat* value)
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform2fv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 3 float values py pointer to the uniform variable "name"
SLint SLGLProgram::uniform3fv(const SLchar*  name,
                              SLsizei        count,
                              const SLfloat* value)
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
                              const SLfloat* value)
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform4fv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 1 int value py pointer to the uniform variable "name"
SLint SLGLProgram::uniform1iv(const SLchar* name,
                              SLsizei       count,
                              const SLint*  value)
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform1iv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 2 int values py pointer to the uniform variable "name"
SLint SLGLProgram::uniform2iv(const SLchar* name,
                              SLsizei       count,
                              const SLint*  value)
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform2iv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 3 int values py pointer to the uniform variable "name"
SLint SLGLProgram::uniform3iv(const SLchar* name,
                              SLsizei       count,
                              const SLint*  value)
{
    SLint loc = getUniformLocation(name);
    if (loc >= 0) glUniform3iv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 4 int values py pointer to the uniform variable "name"
SLint SLGLProgram::uniform4iv(const SLchar* name,
                              SLsizei       count,
                              const SLint*  value)
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
                                    GLboolean      transpose)
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
                                   GLboolean      transpose)
{
    glUniformMatrix2fv(loc, count, transpose, value);
}
//-----------------------------------------------------------------------------
//! Passes a 3x3 float matrix values py pointer to the uniform variable "name"
SLint SLGLProgram::uniformMatrix3fv(const SLchar*  name,
                                    SLsizei        count,
                                    const SLfloat* value,
                                    GLboolean      transpose)
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
                                   GLboolean      transpose)
{
    glUniformMatrix3fv(loc, count, transpose, value);
}
//-----------------------------------------------------------------------------
//! Passes a 4x4 float matrix values py pointer to the uniform variable "name"
SLint SLGLProgram::uniformMatrix4fv(const SLchar*  name,
                                    SLsizei        count,
                                    const SLfloat* value,
                                    GLboolean      transpose)
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
                                   GLboolean      transpose)
{
    glUniformMatrix4fv(loc, count, transpose, value);
}
//-----------------------------------------------------------------------------
