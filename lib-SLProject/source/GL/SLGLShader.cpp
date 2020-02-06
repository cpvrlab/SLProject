//#############################################################################
//  File:      SLGLShader.cpp
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

#include <Utils.h>
#include <SLGLProgram.h>
#include <SLGLShader.h>

//-----------------------------------------------------------------------------
// Error Strings
const SLchar* aGLSLErrorString[] = {(const SLchar*)"(e0000) GLSL not enabled",
                                    (const SLchar*)"(e0001) not a valid program object",
                                    (const SLchar*)"(e0002) not a valid object",
                                    (const SLchar*)"(e0003) out of memory",
                                    (const SLchar*)"(e0004) unknown compiler error"};

//-----------------------------------------------------------------------------
//! Default constructor
SLGLShader::SLGLShader()
{
    _type     = ST_none;
    _code     = "";
    _shaderID = 0;
    _file     = "";
}
//-----------------------------------------------------------------------------
//! Ctor with shader filename & shader type
SLGLShader::SLGLShader(const SLstring& filename, SLShaderType shaderType)
  : SLObject(Utils::getFileName(filename), filename)
{
    _type     = shaderType;
    _code     = "";
    _shaderID = 0;
    _file     = filename;

    // Only load file at this moment, don't compile it.
    load(filename);
}
//-----------------------------------------------------------------------------
//! SLGLShader::load loads a shader file into string _shaderSource
void SLGLShader::load(const SLstring& filename)
{
    fstream shaderFile(filename.c_str(), ios::in);

    if (!shaderFile.is_open())
    {
        SL_LOG("File open failed in SLGLShader::load: %s", filename.c_str());
        exit(1);
    }

    std::stringstream buffer;
    buffer << shaderFile.rdbuf();

    // remove comments because some stupid ARM compiler can't handle GLSL comments
    _code = removeComments(buffer.str());
}
//-----------------------------------------------------------------------------
//! SLGLShader::load loads a shader file from memory into memory
void SLGLShader::loadFromMemory(const SLstring& shaderSource)
{
    _code = shaderSource;
}
//-----------------------------------------------------------------------------
SLGLShader::~SLGLShader()
{
    //SL_LOG("~SLGLShader(%s)", name().c_str());
    if (_shaderID)
        glDeleteShader(_shaderID);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
SLbool SLGLShader::createAndCompileSimple()
{
    // delete if object already exits
    if (_shaderID) glDeleteShader(_shaderID);

    if (!_code.empty())
    {
        switch (_type)
        {
            case ST_vertex:
                _shaderID = glCreateShader(GL_VERTEX_SHADER);
                break;
            case ST_fragment:
                _shaderID = glCreateShader(GL_FRAGMENT_SHADER);
                break;
#if defined(GL_VERSION_4_0) || defined(GL_ES_VERSION_3_2)
            case ST_geometry:
                _shaderID = glCreateShader(GL_GEOMETRY_SHADER);
                break;
#endif
            default:
                SL_EXIT_MSG("SLGLShader::load: Unknown shader type.");
        }
    }

    const char* src = _code.c_str();
    glShaderSource(_shaderID, 1, &src, nullptr);
    glCompileShader(_shaderID);

    // Check compiler log
    SLint compileSuccess = 0;
    glGetShaderiv(_shaderID, GL_COMPILE_STATUS, &compileSuccess);
    if (compileSuccess == GL_FALSE)
    {
        GLchar log[256];
        glGetShaderInfoLog(_shaderID, sizeof(log), nullptr, &log[0]);
        SL_LOG("*** COMPILER ERROR ***");
        SL_LOG("Source file: %s", _file.c_str());
        SL_LOG("%s\n---", log);
        SL_LOG("%s", src);
        return false;
    }
    return true;
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
    if (_shaderID) glDeleteShader(_shaderID);

    if (!_code.empty())
    {
        switch (_type)
        {
            case ST_vertex:
                _shaderID = glCreateShader(GL_VERTEX_SHADER);
                break;
            case ST_fragment:
                _shaderID = glCreateShader(GL_FRAGMENT_SHADER);
                break;
            default:
                SL_EXIT_MSG("SLGLShader::load: Unknown shader type.");
        }

        // Build version string as the first statement
        SLGLState* state      = SLGLState::instance();
        SLstring   verGLSL    = state->glSLVersionNO();
        SLstring   srcVersion = "#version " + verGLSL;
        if (state->glIsES3()) srcVersion += " es";
        srcVersion += "\n";

        // Replace "attribute" and "varying" that came in GLSL 310
        if (verGLSL > "120")
        {
            if (_type == ST_vertex)
            {
                Utils::replaceString(_code, "attribute", "in       ");
                Utils::replaceString(_code, "varying", "out    ");
            }
            if (_type == ST_fragment)
            {
                Utils::replaceString(_code, "varying", "in     ");
            }
        }

        // Replace "gl_FragColor" that was deprecated in GLSL 140 (OpenGL 3.1) by a custom out variable
        if (verGLSL > "130")
        {
            if (_type == ST_fragment)
            {
                Utils::replaceString(_code, "gl_FragColor", "fragColor");
                Utils::replaceString(_code, "void main", "out vec4 fragColor; \n\nvoid main");
            }
        }

        // Replace deprecated texture functions
        if (verGLSL > "140")
        {
            if (_type == ST_fragment)
            {
                Utils::replaceString(_code, "texture1D", "texture");
                Utils::replaceString(_code, "texture2D", "texture");
                Utils::replaceString(_code, "texture3D", "texture");
                Utils::replaceString(_code, "textureCube", "texture");
            }
        }

        _code = srcVersion + _code;

        //// write out the parsed shader code as text files
        //#ifdef _GLDEBUG
        //ofstream fs(name()+".Debug");
        //if(fs)
        //{
        //    fs << _code;
        //    fs.close();
        //}
        //#endif

        const char* src = _code.c_str();
        glShaderSource(_shaderID, 1, &src, nullptr);
        glCompileShader(_shaderID);

        // Check compiler log
        SLint compileSuccess = 0;
        glGetShaderiv(_shaderID, GL_COMPILE_STATUS, &compileSuccess);
        if (compileSuccess == GL_FALSE)
        {
            GLchar log[256];
            glGetShaderInfoLog(_shaderID, sizeof(log), nullptr, &log[0]);
            SL_LOG("*** COMPILER ERROR ***");
            SL_LOG("Source file: %s\n", _file.c_str());
            SL_LOG("%s---", log);
            SL_LOG("%s", src);
            return false;
        }
        return true;
    }
    else
        SL_WARN_MSG("SLGLShader::createAndCompile: Nothing to compile!");
    return false;
}
//-----------------------------------------------------------------------------
//! SLGLShader::removeComments for C/C++ comments removal from shader code
SLstring SLGLShader::removeComments(SLstring src)
{
    SLstring dst;
    SLuint   len    = (SLuint)src.length();
    SLuint   i      = 0;
    SLint    line   = 0;
    SLint    column = 0;

    while (i < len)
    {
        if (src[i] == '/' && src[i + 1] == '/')
        {
            if (column > 0)
                dst += '\n';
            while (i < len && src[i] != '\n')
                i++;
            i++;
        }
        else if (src[i] == '/' && src[i + 1] == '*')
        {
            while (i < len && !(src[i] == '*' && src[i + 1] == '/'))
            {
                if (src[i] == '\n') dst += '\n';
                i++;
            }
            i += 2;
        }
        else
        {
            if (src[i] == '\n')
            {
                line++;
                column = 0;
            }
            else
                column++;

            dst += src[i++];
        }
    }
    //cout << dst << "|" << endl;
    return dst;
}
//-----------------------------------------------------------------------------
//! Returns the shader type as string
SLstring SLGLShader::typeName()
{
    switch (_type)
    {
        case ST_vertex: return "Vertex";
        case ST_fragment: return "Fragment";
        case ST_geometry: return "Geometry";
        case ST_tesselation: return "Tesselation";
        default: return "Unknown";
    }
}
// ----------------------------------------------------------------------------
