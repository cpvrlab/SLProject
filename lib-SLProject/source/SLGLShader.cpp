//#############################################################################
//  File:      SLGLShader.h
//  Author:    Marcus Hudritsch 
//             Mainly based on Martin Christens GLSL Tutorial
//             See http://www.clockworkcoders.com
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include "SLGLShader.h"
#include "SLGLProgram.h"

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
    _type = shaderType;
    _code = "";
    _objectGL = 0;
    _file = filename;
   
    // Only load file at this moment, don't compile it.
    load(filename);
}
//----------------------------------------------------------------------------- 
SLGLShader::~SLGLShader()
{  
    //SL_LOG("~SLGLShader(%s)\n", name().c_str());
    if (_objectGL)
        glDeleteShader(_objectGL);
    GET_GL_ERROR;
}
//----------------------------------------------------------------------------- 
//! SLGLShader::createAndCompile creates & compiles the OpenGL shader object
SLbool SLGLShader::createAndCompile()
{  
    // delete if object already exits
    if (_objectGL) glDeleteShader(_objectGL);

    if (_code!="")
    {  
        switch (_type)
        {   case VertexShader:
                _objectGL = glCreateShader(GL_VERTEX_SHADER); break;
            case FragmentShader:
                _objectGL = glCreateShader(GL_FRAGMENT_SHADER); break;
            default:
                SL_EXIT_MSG("SLGLShader::load: Unknown shader type.");
        }
      
        //SLstring verGLSL = SLGLState::getInstance()->glSLVersionNO();
        //SLstring srcVersion = "#version " + verGLSL + "\n";

        //if (verGLSL > "120")
        //{   if (_type == VertexShader)
        //    {   SLUtils::replaceString(_code, "attribute", "in");
        //        SLUtils::replaceString(_code, "varying", "out");
        //    }
        //    if (_type == FragmentShader)
        //    {   SLUtils::replaceString(_code, "varying", "in");
        //    }
        //}
        //SLstring scrComplete = srcVersion + _code;

        SLstring scrComplete = _code;

        const char* src = scrComplete.c_str();
        glShaderSource(_objectGL, 1, &src, 0);
        glCompileShader(_objectGL);

        // Check compiler log
        SLint compileSuccess = 0;
        glGetShaderiv(_objectGL, GL_COMPILE_STATUS, &compileSuccess);
        if (compileSuccess == GL_FALSE) 
        {   GLchar log[256];
            glGetShaderInfoLog(_objectGL, sizeof(log), 0, &log[0]);
            SL_LOG("*** COMPILER ERROR ***\n");
            SL_LOG("Source file: %s\n", _file.c_str());
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
    #ifdef SL_OS_MACIOS
    _code = buffer.str();
    #else
    _code = SLUtils::removeComments(buffer.str());
    #endif
}
//-----------------------------------------------------------------------------
//! SLGLShader::load loads a shader file from memory into memory 
void SLGLShader::loadFromMemory(const SLstring shaderSource)
{
    _code = shaderSource;
}
// ----------------------------------------------------------------------------
