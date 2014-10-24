//#############################################################################
//  File:      SLGLShaderProg.cpp
//  Author:    Marcus Hudritsch 
//             Mainly based on Martin Christens GLSL Tutorial
//             See http://www.clockworkcoders.com
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLScene.h>
#include <SLGLShaderProg.h>
#include <SLGLShader.h>

//-----------------------------------------------------------------------------
//! Default path for shader files used when only filename is passed in load.
SLstring SLGLShaderProg::defaultPath = "../lib-SLProject/source/oglsl/";
//-----------------------------------------------------------------------------
// Error Strings defined in SLGLShader.h
extern char* aGLSLErrorString[];
//-----------------------------------------------------------------------------
//! Ctor with a vertex and a fragment shader filename
SLGLShaderProg::SLGLShaderProg(SLstring vertShaderFile, 
                               SLstring fragShaderFile) : SLObject("")
{  
    _stateGL = SLGLState::getInstance();
    _isLinked = false;
    _programObjectGL = 0;   

    // optional load vertex and/or fragment shaders
    addShader(new SLGLShader(defaultPath+vertShaderFile, VertexShader));
    addShader(new SLGLShader(defaultPath+fragShaderFile, FragmentShader));

    // Add pointer to the global resource vectors for deallocation
    SLScene::current->shaderProgs().push_back(this);
}
//-----------------------------------------------------------------------------
//! The destructor detaches all shader objects and deletes them
SLGLShaderProg::~SLGLShaderProg()
{  
    //SL_LOG("~SLGLShaderProg\n");
      
    for (SLuint i=0; i<_shaderList.size(); i++)
    {   if (_isLinked)
        {   glDetachShader(_programObjectGL, _shaderList[i]->_shaderObjectGL);
            GET_GL_ERROR;
        }
      
        // always delete shader objects before program object
        delete _shaderList[i]; 
    }                      

    if (_programObjectGL>0)
    {   glDeleteProgram(_programObjectGL);
        GET_GL_ERROR;
    }

    // delete uniform variables
    for (SLuint i=0; i < _uniform1fList.size(); ++i) delete _uniform1fList[i];
    for (SLuint i=0; i < _uniform1iList.size(); ++i) delete _uniform1iList[i];
}
//-----------------------------------------------------------------------------
//! SLGLShaderProg::addShader adds a shader to the shader list
void SLGLShaderProg::addShader(SLGLShader* shader)
{
    assert(shader);
   _shaderList.push_back(shader); 
}
//-----------------------------------------------------------------------------
/*! SLGLShaderProg::init creates the OpenGL shaderprogram object, compiles all
shader objects and attaches them to the shaderprogram. At the end all shaders
are linked. If a shader fails to compile a simple texture only shader is
compiled that shows an error message in the texture.
*/
void SLGLShaderProg::init()
{     
    // create program object if it doesn't exist
    if(!_programObjectGL) _programObjectGL = glCreateProgram();
   
    // if already linked, detach, recreate and compile shaders
    if (_isLinked)
    {   for (SLuint i=0; i<_shaderList.size(); i++)
        {   if (_isLinked)
            {   glDetachShader(_programObjectGL, _shaderList[i]->_shaderObjectGL);
                GET_GL_ERROR;
            }
        }
        _isLinked = false;
    }
   
    // compile all shader objects
    SLbool allSuccuessfullyCompiled = true;
    for (SLuint i=0; i<_shaderList.size(); i++)
    {   if (!_shaderList[i]->createAndCompile())
        {   allSuccuessfullyCompiled = false;
            break;
        }
        GET_GL_ERROR;
    }

    // try to compile alternative per vertex lighting shaders
    if (!allSuccuessfullyCompiled)
    {        
        // delete all shaders and uniforms that where attached
        for (SLuint i=0; i<_shaderList.size();    i++) delete _shaderList[i]; 
        for (SLuint i=0; i<_uniform1fList.size(); ++i) delete _uniform1fList[i];
        for (SLuint i=0; i<_uniform1iList.size(); ++i) delete _uniform1iList[i];
        _shaderList.clear();
        _uniform1fList.clear();
        _uniform1iList.clear();

        addShader(new SLGLShader(defaultPath+"ErrorTex.vert", VertexShader));
        addShader(new SLGLShader(defaultPath+"ErrorTex.frag", FragmentShader));

        allSuccuessfullyCompiled = true;
        for (SLuint i=0; i<_shaderList.size(); i++)
        {   if (!_shaderList[i]->createAndCompile())
            {  allSuccuessfullyCompiled = false;
                break;
            }
            GET_GL_ERROR;
        }
    }

    // attach all shader objects
    if (allSuccuessfullyCompiled)
    {   for (SLuint i=0; i<_shaderList.size(); i++)
        {   glAttachShader(_programObjectGL, _shaderList[i]->_shaderObjectGL);
            GET_GL_ERROR;
        }
    } else SL_EXIT_MSG("No successufully compiled shaders attached!");
    
    int linked;
    glLinkProgram(_programObjectGL);
    GET_GL_ERROR;
    glGetProgramiv(_programObjectGL, GL_LINK_STATUS, &linked);
    GET_GL_ERROR;

    if (linked)
    {   _isLinked = true;
        for (SLuint i=0; i<_shaderList.size(); i++) 
            _name += "+"+_shaderList[i]->name();
        //SL_LOG("Linked: %s", _name.c_str());
    } else
    {   SLchar log[256];
        glGetProgramInfoLog(_programObjectGL, sizeof(log), 0, &log[0]);
        SL_LOG("*** LINKER ERROR ***\n");
        SL_LOG("Source files: \n");
        for (SLuint i=0; i<_shaderList.size(); i++) 
            SL_LOG("%s\n", _shaderList[i]->name().c_str());
        SL_LOG("%s\n", log);
        SL_EXIT_MSG("GLSL linker error");
    }
}
//-----------------------------------------------------------------------------
/*! SLGLShaderProg::useProgram inits the first time the program and then uses it.
Call this initialization if you pass your own custom uniform variables.
*/
void SLGLShaderProg::useProgram()
{  
    if (_programObjectGL==0 && _shaderList.size()>0) init();

    if (_isLinked)
    {   _stateGL->useProgram(_programObjectGL);
        GET_GL_ERROR;
    }
}
//-----------------------------------------------------------------------------
/*! SLGLShaderProg::beginUse starts using the shaderprogram and transfers the 
the standard light and material parameter as uniform variables. It also passes 
the custom uniform variables of the _uniform1fList as well as the texture names.
*/
void SLGLShaderProg::beginUse(SLMaterial* mat)
{  
    if (_programObjectGL==0 && _shaderList.size()>0) init();

    if (_isLinked)
    {  
        // 1: Activate the shader program object
        _stateGL->useProgram(_programObjectGL);
            
        // 2: Pass light & material parameters
        _stateGL->globalAmbientLight = SLScene::current->globalAmbiLight();
        SLint loc = uniform4fv("u_globalAmbient",  1,  (SLfloat*) _stateGL->globalAmbient());
        loc = uniform1i("u_numLightsUsed", _stateGL->numLightsUsed);

        if (_stateGL->numLightsUsed > 0)
        {   SLint nL = SL_MAX_LIGHTS;
            _stateGL->calcLightPosVS(_stateGL->numLightsUsed);
            _stateGL->calcLightDirVS(_stateGL->numLightsUsed);
            loc = uniform1iv("u_lightIsOn",      nL, (SLint*)   _stateGL->lightIsOn);
            loc = uniform4fv("u_lightPosVS",     nL, (SLfloat*) _stateGL->lightPosVS);
            loc = uniform4fv("u_lightAmbient",   nL, (SLfloat*) _stateGL->lightAmbient);
            loc = uniform4fv("u_lightDiffuse",   nL, (SLfloat*) _stateGL->lightDiffuse);
            loc = uniform4fv("u_lightSpecular",  nL, (SLfloat*) _stateGL->lightSpecular);
            loc = uniform3fv("u_lightDirVS",     nL, (SLfloat*) _stateGL->lightDirVS);
            loc = uniform1fv("u_lightSpotCutoff",nL, (SLfloat*) _stateGL->lightSpotCutoff);
            loc = uniform1fv("u_lightSpotCosCut",nL, (SLfloat*) _stateGL->lightSpotCosCut);
            loc = uniform1fv("u_lightSpotExp",   nL, (SLfloat*) _stateGL->lightSpotExp);
            loc = uniform3fv("u_lightAtt",       nL, (SLfloat*) _stateGL->lightAtt);
            loc = uniform1iv("u_lightDoAtt",     nL, (SLint*)   _stateGL->lightDoAtt);
            loc = uniform4fv("u_matAmbient",     1,  (SLfloat*)&_stateGL->matAmbient);
            loc = uniform4fv("u_matDiffuse",     1,  (SLfloat*)&_stateGL->matDiffuse);
            loc = uniform4fv("u_matSpecular",    1,  (SLfloat*)&_stateGL->matSpecular);
            loc = uniform4fv("u_matEmissive",    1,  (SLfloat*)&_stateGL->matEmissive);
            loc = uniform1f ("u_matShininess",                  _stateGL->matShininess);
        }
      
        // 2b: Set stereo states
        loc = uniform1i ("u_projection", _stateGL->projection);
        loc = uniform1i ("u_stereoEye",  _stateGL->stereoEye);
        loc = uniformMatrix3fv("u_stereoColorFilter", 1, 
                                (SLfloat*)&_stateGL->stereoColorFilter);

        // 2c: Pass diffuse color for uniform color shader
        loc = uniform4fv("u_color", 1,  (SLfloat*)&_stateGL->matDiffuse);

        // 3: Pass the custom uniform1f variables of the list
        for (SLuint i=0; i<_uniform1fList.size(); i++)
            loc = uniform1f(_uniform1fList[i]->name(), _uniform1fList[i]->value());
        for (SLuint i=0; i<_uniform1iList.size(); i++)
            loc = uniform1i(_uniform1iList[i]->name(), _uniform1iList[i]->value());
      
        // 4: Send texture units as uniforms texture samplers
        if (mat)
        {   for (SLint i=0; i<(SLint)mat->textures().size(); ++i)
            {   SLchar name[100];
                sprintf(name,"u_texture%d", i);
                loc = uniform1i(name, i);
            }
        }
        GET_GL_ERROR;
    }
}
//----------------------------------------------------------------------------- 
//! SLGLShaderProg::endUse stops the shaderprogram
void SLGLShaderProg::endUse()
{
    _stateGL->useProgram(0);
    GET_GL_ERROR;
}
//----------------------------------------------------------------------------- 
//! SLGLShaderProg::addUniform1f add a uniform variable to the list
void SLGLShaderProg::addUniform1f(SLGLShaderUniform1f *u)
{
    _uniform1fList.push_back(u);
}
//----------------------------------------------------------------------------- 
//! SLGLShaderProg::addUniform1f add a uniform variable to the list
void SLGLShaderProg::addUniform1i(SLGLShaderUniform1i *u)
{
    _uniform1iList.push_back(u);
}
//-----------------------------------------------------------------------------
/*! SLGLShaderProg::getUniformLocation return the location id of a uniform
variable. For not querying this with OpenGLs glGetUniformLocation we put all
uniform locations into a hash map. glGet* function should be called only once
during shader initialization and not during frame rendering because glGet*
function force a pipeline flush.
*/
//SLint SLGLShaderProg::getUniformLocation(const SLchar *name)
//{  SLint loc;
//   SLLocMap::iterator it = _uniformLocHash.find(name);
//   if(it == _uniformLocHash.end())
//   {  // If not found query it from GL and add it to the hash map
//      loc = glGetUniformLocation(_programObjectGL, name);
//      _uniformLocHash[name] = loc;
//   } else
//      loc = it->second;
//   return loc;
//}
//-----------------------------------------------------------------------------
/*! SLGLShaderProg::getAttribLocation return the location id of a attribute
variable. For not querying this with OpenGLs glGetAttribLocation we put all
attribute locations into a hash map. glGet* function should be called only once
during shader initialization and not during frame rendering because glGet*
function force a pipeline flush.
*/
//SLint SLGLShaderProg::getAttribLocation(const SLchar *name)
//{  SLint loc;
//   SLLocMap::iterator it = _attribLocHash.find(name);
//   if(it == _attribLocHash.end())
//   {  // If not found query it from GL and add it to the hash map
//      loc = glGetAttribLocation(_programObjectGL, name);
//      _attribLocHash[name] = loc;
//   } else
//      loc = it->second;
//   return loc;
//}

//-----------------------------------------------------------------------------
//! Passes the float value v0 to the uniform variable "name"
SLint SLGLShaderProg::uniform1f(const SLchar* name, SLfloat v0)
{
    SLint loc = getUniformLocation(name);
    if (loc>=0) glUniform1f(loc, v0);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the float values v0 & v1 to the uniform variable "name"
SLint SLGLShaderProg::uniform2f(const SLchar* name, SLfloat v0, SLfloat v1)
{
    SLint loc = getUniformLocation(name);
    if (loc>=0) glUniform2f(loc, v0, v1);
    return loc;
}
//----------------------------------------------------------------------------- 
//! Passes the float values v0, v1 & v2 to the uniform variable "name"
SLint SLGLShaderProg::uniform3f(const SLchar* name,
                                SLfloat v0, SLfloat v1, SLfloat v2)
{
    SLint loc = getUniformLocation(name);
    if (loc>=0) glUniform3f(loc, v0, v1, v2);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the float values v0,v1,v2 & v3 to the uniform variable "name"
SLint SLGLShaderProg::uniform4f(const SLchar* name,
                                SLfloat v0, SLfloat v1, SLfloat v2, SLfloat v3)
{
    SLint loc = getUniformLocation(name);
    if (loc>=0) glUniform4f(loc, v0, v1, v2, v3);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the int values v0 to the uniform variable "name"
SLint SLGLShaderProg::uniform1i(const SLchar* name, SLint v0)
{
    SLint loc = getUniformLocation(name);
    if (loc>=0) glUniform1i(loc, v0);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the int values v0 & v1 to the uniform variable "name"
SLint SLGLShaderProg::uniform2i(const SLchar* name, SLint v0, SLint v1)
{
    SLint loc = getUniformLocation(name);
    if (loc>=0) glUniform2i(loc, v0, v1);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the int values v0, v1 & v2 to the uniform variable "name"
SLint SLGLShaderProg::uniform3i(const SLchar* name, SLint v0, SLint v1, SLint v2)
{
    SLint loc = getUniformLocation(name);
    if (loc>=0) glUniform3i(loc, v0, v1, v2);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes the int values v0, v1, v2 & v3 to the uniform variable "name"
SLint SLGLShaderProg::uniform4i(const SLchar* name, SLint v0, SLint v1, SLint v2, 
                               SLint v3)
{
    SLint loc = getUniformLocation(name);
    if (loc==-1) return false;
    glUniform4i(loc, v0, v1, v2, v3);
    return loc;
}
//----------------------------------------------------------------------------- 
//! Passes 1 float value py pointer to the uniform variable "name"
SLint SLGLShaderProg::uniform1fv(const SLchar* name, 
                                 SLsizei count, const SLfloat* value)
{
    SLint loc = getUniformLocation(name);
    if (loc>=0) glUniform1fv(loc, count, value);
    return loc;
}
//----------------------------------------------------------------------------- 
//! Passes 2 float values py pointer to the uniform variable "name"
SLint SLGLShaderProg::uniform2fv(const SLchar* name,
                                 SLsizei count, const SLfloat* value)
{
    SLint loc = getUniformLocation(name);
    if (loc>=0) glUniform2fv(loc, count, value);
    return loc;
}
//----------------------------------------------------------------------------- 
//! Passes 3 float values py pointer to the uniform variable "name"
SLint SLGLShaderProg::uniform3fv(const SLchar* name,
                                 SLsizei count, const SLfloat* value)
{
    SLint loc = getUniformLocation(name);
    if (loc==-1) return false;
    glUniform3fv(loc, count, value);
    return loc;
}
//----------------------------------------------------------------------------- 
//! Passes 4 float values py pointer to the uniform variable "name"
SLint SLGLShaderProg::uniform4fv(const SLchar* name,
                                 SLsizei count, const SLfloat* value)
{
    SLint loc = getUniformLocation(name);
    if (loc>=0) glUniform4fv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 1 int value py pointer to the uniform variable "name" 
SLint SLGLShaderProg::uniform1iv(const SLchar* name, 
                                 SLsizei count, const SLint* value)
{
    SLint loc = getUniformLocation(name);
    if (loc>=0) glUniform1iv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 2 int values py pointer to the uniform variable "name"  
SLint SLGLShaderProg::uniform2iv(const SLchar* name,
                                 SLsizei count, const SLint* value)
{
    SLint loc = getUniformLocation(name);
    if (loc>=0) glUniform2iv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 3 int values py pointer to the uniform variable "name"   
SLint SLGLShaderProg::uniform3iv(const SLchar* name, 
                                 SLsizei count, const SLint* value)
{
    SLint loc = getUniformLocation(name);
    if (loc>=0) glUniform3iv(loc, count, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes 4 int values py pointer to the uniform variable "name"   
SLint SLGLShaderProg::uniform4iv(const SLchar* name, 
                                 SLsizei count, const SLint* value)
{
    SLint loc = getUniformLocation(name);
    if (loc>=0) glUniform4iv(loc, count, value);
    return loc;
}
//----------------------------------------------------------------------------- 
//! Passes a 2x2 float matrix values py pointer to the uniform variable "name"  
SLint SLGLShaderProg::uniformMatrix2fv(const SLchar* name, SLsizei count, 
                                       const SLfloat* value, GLboolean transpose)
{
    SLint loc = getUniformLocation(name);
    if (loc>=0) glUniformMatrix2fv(loc, count, transpose, value);
    return loc;
}
//----------------------------------------------------------------------------- 
//! Passes a 2x2 float matrix values py pointer to the uniform at location loc  
void SLGLShaderProg::uniformMatrix2fv(const SLint loc, SLsizei count, 
                                      const SLfloat* value, GLboolean transpose)
{
    glUniformMatrix2fv(loc, count, transpose, value);
}
//-----------------------------------------------------------------------------
//! Passes a 3x3 float matrix values py pointer to the uniform variable "name"   
SLint SLGLShaderProg::uniformMatrix3fv(const SLchar* name, SLsizei count, 
                                       const SLfloat* value, GLboolean transpose)
{
    SLint loc = getUniformLocation(name);
    if (loc>=0) glUniformMatrix3fv(loc, count, transpose, value);
    return loc;
}
//-----------------------------------------------------------------------------
//! Passes a 3x3 float matrix values py pointer to the uniform at location loc    
void SLGLShaderProg::uniformMatrix3fv(const SLint loc, SLsizei count, 
                                      const SLfloat* value, GLboolean transpose)
{
    glUniformMatrix3fv(loc, count, transpose, value);
}
//----------------------------------------------------------------------------- 
//! Passes a 4x4 float matrix values py pointer to the uniform variable "name"  
SLint SLGLShaderProg::uniformMatrix4fv(const SLchar* name, SLsizei count, 
                                       const SLfloat* value, GLboolean transpose)
{
    SLint loc = getUniformLocation(name);
    if (loc>=0) glUniformMatrix4fv(loc, count, transpose, value);
    return loc;
}
//----------------------------------------------------------------------------- 
//! Passes a 4x4 float matrix values py pointer to the uniform at location loc 
void SLGLShaderProg::uniformMatrix4fv(const SLint loc, SLsizei count, 
                                      const SLfloat* value, GLboolean transpose)
{
    glUniformMatrix4fv(loc, count, transpose, value);
}
//----------------------------------------------------------------------------- 

