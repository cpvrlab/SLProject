//#############################################################################
//  File:      SLGLShader.cpp
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <Utils.h>
#include <SLGLState.h>
#include <SLGLProgram.h>
#include <SLGLShader.h>
#include "SLFileStorage.h"

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
/*! If the shader filename does not belong to an existing file the shader code
 * will be generated at a later stage by SLGLProgramGenerated.
 * @param filename Path and filename of the shader to be loaded or generated.
 * @param shaderType Shader type: ST_vertex, ST_fragment and ST_geometry.
 */
SLGLShader::SLGLShader(const SLstring& filename, SLShaderType shaderType)
  : SLObject(Utils::getFileName(filename), filename)
{
    _type     = shaderType;
    _code     = "";
    _shaderID = 0;
    _file     = filename;

    // Only load file at this moment, don't compile it.
    if (SLFileStorage::exists(filename, IOK_shader))
        load(filename);
}
//-----------------------------------------------------------------------------
//! SLGLShader::load loads a shader file into string _shaderSource
void SLGLShader::load(const SLstring& filename)
{
    SLIOBuffer buffer = SLFileStorage::readIntoBuffer(filename, IOK_shader);
    _code             = SLstring(buffer.data, buffer.data + buffer.size);
    SLFileStorage::deleteBuffer(buffer);

    // remove comments because some stupid ARM compiler can't handle GLSL comments
    _code = removeComments(_code);
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
    // SL_LOG("~SLGLShader(%s)", name().c_str());
    if (_shaderID)
        glDeleteShader(_shaderID);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
//! SLGLShader::createAndCompile creates & compiles the OpenGL shader object
/*!
@return true if compilation was successful
*/
SLbool SLGLShader::createAndCompile(SLVLight* lights)
{
    // delete if object already exits
    if (_shaderID)
        glDeleteShader(_shaderID);

    if (_code.empty())
    {
        SL_WARN_MSG("SLGLShader::createAndCompile: Nothing to compile!");
        std::cout << "file: " << _file << std::endl;
        return false;
    }

    switch (_type)
    {
        case ST_vertex:
            _shaderID = glCreateShader(GL_VERTEX_SHADER);
            break;
        case ST_geometry:
            /*
             * 0x8DD9 == GL_GEOMETRY_SHADER would be defined in <GLES3/gl32.h>
             * So far this header is not included by default in Android SDK and therefore
             * it is not guaranteed that an Android device hat OpenGL ES 3.2 and this header.
             * The app knows the OpenGL version when it is running. App that are below OpenGL ES 3.2
             * should not try to create geometry shaders as used for particles.
             */
            _shaderID = glCreateShader(0x8DD9);
            break;
        case ST_fragment:
            _shaderID = glCreateShader(GL_FRAGMENT_SHADER);
            break;
        default:
            SL_EXIT_MSG("SLGLShader::load: Unknown shader type.");
    }
    GET_GL_ERROR;

    // Build version string as the first statement
    SLGLState* state      = SLGLState::instance();
    SLstring   verGLSL    = state->glSLVersionNO();
    SLstring   srcVersion = "#version " + verGLSL;
    if (state->glIsES3()) srcVersion += " es";
    srcVersion += "\n";

    _code = preprocessPragmas(_code, lights);

    // Concatenate final code string
    _code = srcVersion + _code;

    const char* src = _code.c_str();
    glShaderSource(_shaderID, 1, &src, nullptr);
    GET_GL_ERROR;

    glCompileShader(_shaderID);
    GET_GL_ERROR;

    // Check compiler log
    SLint compileSuccess = 0;
    glGetShaderiv(_shaderID, GL_COMPILE_STATUS, &compileSuccess);
    GET_GL_ERROR;
    if (compileSuccess == GL_FALSE)
    {
        GLchar log[1024];
        glGetShaderInfoLog(_shaderID, sizeof(log), nullptr, &log[0]);
        SL_LOG("*** COMPILER ERROR ***");
        SL_LOG("Source file: %s\n", _file.c_str());
        SL_LOG("%s---", log);
        SLVstring lines   = Utils::getStringLines(_code);
        SLint     lineNum = 1;
        for (string& line : lines)
            SL_LOG("%4d: %s", lineNum++, line.c_str());
        return false;
    }

    // Write generated shader out
    if (!_file.empty())
    {
#if defined(DEBUG) || defined(_DEBUG)
        string filename = Utils::getFileName(_file);
        string path     = Utils::getDirName(_file);
        if (Utils::dirExists(path))
        {
            if (Utils::containsString(path, "generatedShaders"))
            {
                SLFileStorage::writeString(_file, IOK_shader, _code);
                SL_LOG("Exported Shader Program: %s", filename.c_str());
            }
        }
        else
            SL_WARN_MSG("**** No path to write shader ***");
#else
        if (!SLFileStorage::exists(_file, IOK_shader))
        {
            string filename = Utils::getFileName(_file);
            string path     = Utils::getDirName(_file);
            if (Utils::dirExists(path))
            {
                SLFileStorage::writeString(_file, IOK_shader, _code);
                SL_LOG("Exported Shader Program: %s", filename.c_str());
            }
            else
                SL_WARN_MSG("**** No path to write shader ***");
        }
#endif
    }
    else
        SL_WARN_MSG("**** No shader path and filename for shader ***");

    return true;
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
//! Replaces our own pragma directives in GLSL code
SLstring SLGLShader::preprocessPragmas(SLstring inCode, SLVLight* lights)
{
    // Check first if #pragma exists at all
    size_t pragmaStart = inCode.find("#pragma");
    if (pragmaStart == string::npos)
        return inCode;

    SLVstring codeLines = Utils::getStringLines(inCode);

    string outCode;

    for (string& line : codeLines)
    {
        pragmaStart = line.find("#pragma");
        if (pragmaStart == string::npos)
            outCode += line + '\n';
        else
        {
            SLVstring pragmaParts;
            Utils::splitString(line, ' ', pragmaParts);

            for (auto& part : pragmaParts)
                part = Utils::trimString(part);

            if (pragmaParts[1] == "include") //................................
            {
                string filename = Utils::trimString(pragmaParts[2], "\"");
                string path     = Utils::getPath(_file);
                string pathFile = path + filename;
                if (SLFileStorage::exists(pathFile, IOK_shader))
                {
                    string includeCode = SLFileStorage::readIntoString(pathFile, IOK_shader);
                    includeCode        = removeComments(includeCode);
                    outCode += includeCode + '\n';
                }
                else
                {
                    SL_LOG("SLGLShader::preprocessPragmas: File doesn't exist: %s",
                           pathFile.c_str());
                    outCode += line + '\n';
                }
            }
            else if (pragmaParts[1] == "define") //............................
            {
                if (pragmaParts[2] == "NUM_LIGHTS")
                {
                    outCode += "#define NUM_LIGHTS " +
                               std::to_string(lights->size()) + "\n";
                }
                else
                    outCode += line + '\n';
            } //...............................................................
            else
                outCode += line + '\n';
        }
    }
    return outCode;
}
//-----------------------------------------------------------------------------
