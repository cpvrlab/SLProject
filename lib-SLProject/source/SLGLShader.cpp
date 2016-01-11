//#############################################################################
//  File:      SLGLShader.cpp
//  Author:    Marcus Hudritsch
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
    _code = removeComments(buffer.str());
    #endif
}
//-----------------------------------------------------------------------------
//! SLGLShader::load loads a shader file from memory into memory 
void SLGLShader::loadFromMemory(const SLstring shaderSource)
{
    _code = shaderSource;
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
/*!
All shaders are written with the initial GLSL version 110 and are therefore
backwards compatible with the compatibility profile from OpenGL 2.1 and
OpenGL ES 2 that runs on most mobile devices. To be upwards compatible some
modification have to be done.
\return true if compilation was successfull
*/
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

        SLstring verGLSL = SLGLState::getInstance()->glSLVersionNO();
        SLstring srcVersion = "#version " + verGLSL + "\n";

        // Replace "attribute" and "varying" that came in GLSL 310
        if (verGLSL > "120")
        {   if (_type == VertexShader)
            {   SLUtils::replaceString(_code, "attribute", "in       ");
                SLUtils::replaceString(_code, "varying", "out    ");
            }
            if (_type == FragmentShader)
            {   SLUtils::replaceString(_code, "varying", "in     ");
            }
        }

        // Replace "gl_FragColor" that was deprecated in GLSL 140 (OpenGL 3.1) by a custom out variable
        if (verGLSL > "130")
        {
            if (_type == FragmentShader)
            {   SLUtils::replaceString(_code, "gl_FragColor", "fragColor");
                SLUtils::replaceString(_code, "void main", "out vec4 fragColor; \n\nvoid main");
            }
        }     
        
        // Replace deprecated texture functions
        if (verGLSL > "140")
        {   if (_type == FragmentShader)
            {   SLUtils::replaceString(_code, "texture2D", "texture");
                SLUtils::replaceString(_code, "texture3D", "texture");
                SLUtils::replaceString(_code, "textureCube", "texture");
            }
        }
        
        SLstring scrComplete = srcVersion + _code;
        
        // write out the parsed shader code as text files
        #ifdef _GLDEBUG
        ofstream fs(name()+".Debug"); 
        if(fs)
        {
            fs << scrComplete;
            fs.close();
        }
        #endif

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
//! SLUtils::removeComments for C/C++ comments removal from shader code
SLstring SLGLShader::removeComments(SLstring src)
{  
    SLstring dst;
    SLint len = (SLint)src.length();
    SLint i = 0;
    SLint line = 0;
    SLint column = 0;

    while (i < len)
    {   if (src[i]=='/' && src[i+1]=='/')
        {   if (column > 0)
                dst += '\n';
            while (i<len && src[i] != '\n') i++;
            i++; 
        } 
        else if (src[i]=='/' && src[i+1]=='*')
        {   while (i<len && !(src[i]=='*' && src[i+1]=='/'))
            {   if (src[i]=='\n') dst += '\n';
                i++; 
            }
            i+=2;
        } 
        else
        {   if (src[i] == '\n')
            {   line++;
                column = 0;
            } else column++;

            dst += src[i++];
        } 
    }
    //cout << dst << "|" << endl;
    return dst;
}
// ----------------------------------------------------------------------------
