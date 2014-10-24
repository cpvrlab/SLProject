//#############################################################################
//  File:      SLGLShader.h
//  Author:    Marcus Hudritsch 
//             Mainly based on Martin Christens GLSL Tutorial
//             See http://www.clockworkcoders.com
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include "SLGLShader.h"
#include "SLGLShaderProg.h"

//-----------------------------------------------------------------------------
// Error Strings
SLchar* aGLSLErrorString[] = {(SLchar*)"(e0000) GLSL not enabled",
                              (SLchar*)"(e0001) not a valid program object",
                              (SLchar*)"(e0002) not a valid object",
                              (SLchar*)"(e0003) out of memory",
                              (SLchar*)"(e0004) unknown compiler error"};
                              
//----------------------------------------------------------------------------- 
//! Ctor with shader filename & shader type
SLGLShader::SLGLShader(SLstring filename, SLShaderType shaderType) 
           :SLObject(SLUtils::getFileName(filename))
{  
    _shaderType = shaderType;
    _shaderSource = "";
    _shaderObjectGL = 0;
    _shaderFile = filename;
   
    // Only load file at this moment, don't compile.
    load(filename);
}
//----------------------------------------------------------------------------- 
SLGLShader::~SLGLShader()
{  
    //SL_LOG("~SLGLShader(%s)\n", name().c_str());
    if (_shaderObjectGL)  
        glDeleteShader(_shaderObjectGL);
    GET_GL_ERROR;
}
//----------------------------------------------------------------------------- 
//! SLGLShader::createAndCompile creates & compiles the OpenGL shader object
SLbool SLGLShader::createAndCompile()
{  
    // delete if object already exits
    if (_shaderObjectGL) glDeleteShader(_shaderObjectGL);

    if (_shaderSource!="")
    {  
        switch (_shaderType)
        {   case VertexShader:
                _shaderObjectGL = glCreateShader(GL_VERTEX_SHADER); break;
            case FragmentShader:
                _shaderObjectGL = glCreateShader(GL_FRAGMENT_SHADER); break;
            default:
                SL_EXIT_MSG("SLGLShader::load: Unknown shader type.");
        }
      
        SLstring srcVersion = "";
        SLstring scrDefines = "";


        SLstring scrComplete = srcVersion + 
                               scrDefines + 
                               _shaderSource;

        const char* src = scrComplete.c_str();
        glShaderSource(_shaderObjectGL, 1, &src, 0);   
        glCompileShader(_shaderObjectGL);   

        // Check comiler log
        SLint compileSuccess = 0;
        glGetShaderiv(_shaderObjectGL, GL_COMPILE_STATUS, &compileSuccess);
        if (compileSuccess == GL_FALSE) 
        {   GLchar log[256];
            glGetShaderInfoLog(_shaderObjectGL, sizeof(log), 0, &log[0]);
            SL_LOG("*** COMPILER ERROR ***\n");
            SL_LOG("Source file: %s\n", _shaderFile.c_str());
            //SL_LOG("Source: %s\n", _shaderSource.c_str());
            SL_LOG("%s\n\n", log);
            return false;
        }
        return true;
    } else SL_WARN_MSG("SLGLShader::createAndCompile: Nothing to compile!");
    return false;
}
//-----------------------------------------------------------------------------
//! SLGLShader::load loads a shader file into string _shaderSource
void SLGLShader::load(SLstring filename)
{  
    fstream shaderFile(filename.c_str(), ios::in);
    
    if (!shaderFile.is_open())
    {   SL_LOG("File open failed: %s\n", filename.c_str());
        exit(1);
    }
   
    std::stringstream buffer;
    buffer << shaderFile.rdbuf(); 

    // remove comments because some stupid ARM compiler can't handle GLSL comments
    _shaderSource = SLUtils::removeComments(buffer.str());
}
//-----------------------------------------------------------------------------
//! SLGLShader::load loads a shader file from memory into memory 
void SLGLShader::loadFromMemory(const SLstring shaderSource)
{
    _shaderSource = shaderSource;
}
// ----------------------------------------------------------------------------
