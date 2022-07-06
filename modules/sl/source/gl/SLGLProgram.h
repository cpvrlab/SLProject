//#############################################################################
//  File:      SLGLProgram.h
//  Authors:   Marcus Hudritsch
//             Mainly based on Martin Christens GLSL Tutorial
//             See http://www.clockworkcoders.com
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLPROGRAM_H
#define SLGLPROGRAM_H

#include <map>

#include <SLGLState.h>
#include <SLVec4.h>
#include <SLGLUniform.h>
#include <SLObject.h>
#include <SLLight.h>

class SLGLShader;
class SLScene;
class SLMaterial;
class SLGLState;
class SLAssetManager;
class SLCamera;
class SLSkybox;

//-----------------------------------------------------------------------------
//! STL vector type for SLGLShader pointers
typedef vector<SLGLShader*> SLVGLShader;

#if defined(TARGET_OS_IOS)
// The TR1 unordered_map or the hash_map is not yet available on iOS
typedef std::map<string, int> SLLocMap;
#else
// typedef unordered_map<const char*, int, hash<const char*>, eqstr> SLLocMap;
typedef std::map<string, int> SLLocMap;
#endif

//-----------------------------------------------------------------------------
//! Encapsulation of an OpenGL shader program object
/*!
The SLGLProgram base class represents a shader program object of the OpenGL
Shading Language (GLSL). Multiple SLGLShader objects can be attached and linked
at run time. An SLGLProgram object can then be attached to an SLMaterial
node for execution. An SLGLProgram object can hold an vector of uniform
variable that can transfer variables from the CPU program to the GPU program.
For more details on GLSL please refer to official GLSL documentation and to
SLGLShader.<br>
All shader files are located in the directory data/shaders. For OSX, iOS and
Android applications they are copied to the appropriate file system locations.
*/
//-----------------------------------------------------------------------------
class SLGLProgram : public SLObject
{
public:
    SLGLProgram(SLAssetManager* am,
                const string&   vertShaderFile,
                const string&   fragShaderFile,
                const string&   geomShaderFile = "",
                const string&   programName    = "");

    ~SLGLProgram() override;

    void deleteDataGpu();
    void addShader(SLGLShader* shader);
    void init(SLVLight* lights);
    void initTF(const char* writeBackAttrib[], int size);

    virtual void beginShader(SLCamera*   cam,
                             SLMaterial* mat,
                             SLVLight*   lights) = 0; //!< starter for derived classes
    virtual void endShader()                   = 0;

    void  beginUse(SLCamera*   cam,
                   SLMaterial* mat,
                   SLVLight*   lights);
    SLint passLightsToUniforms(SLVLight* lights,
                               SLuint    nextTexUnit) const;
    void  endUse();
    void  useProgram();

    void addUniform1f(SLGLUniform1f* u); //!< add float uniform
    void addUniform1i(SLGLUniform1i* u); //!< add int uniform

    // Getters
    SLuint       progID() const { return _progID; }
    SLVGLShader& shaders() { return _shaders; }

    // Variable location getters
    SLint getUniformLocation(const SLchar* name) const;

    // Send uniform variables to program
    SLint uniform1f(const SLchar* name, SLfloat v0) const;
    SLint uniform2f(const SLchar* name, SLfloat v0, SLfloat v1) const;
    SLint uniform3f(const SLchar* name, SLfloat v0, SLfloat v1, SLfloat v2) const;
    SLint uniform4f(const SLchar* name, SLfloat v0, SLfloat v1, SLfloat v2, SLfloat v3) const;

    SLint uniform1i(const SLchar* name, SLint v0) const;
    SLint uniform2i(const SLchar* name, SLint v0, SLint v1) const;
    SLint uniform3i(const SLchar* name, SLint v0, SLint v1, SLint v2) const;
    SLint uniform4i(const SLchar* name, SLint v0, SLint v1, SLint v2, SLint v3) const;

    SLint uniform1fv(const SLchar* name, SLsizei count, const SLfloat* value) const;
    SLint uniform2fv(const SLchar* name, SLsizei count, const SLfloat* value) const;
    SLint uniform3fv(const SLchar* name, SLsizei count, const SLfloat* value) const;
    SLint uniform4fv(const SLchar* name, SLsizei count, const SLfloat* value) const;

    SLint uniform1iv(const SLchar* name, SLsizei count, const SLint* value) const;
    SLint uniform2iv(const SLchar* name, SLsizei count, const SLint* value) const;
    SLint uniform3iv(const SLchar* name, SLsizei count, const SLint* value) const;
    SLint uniform4iv(const SLchar* name, GLsizei count, const SLint* value) const;

    SLint uniformMatrix2fv(const SLchar*  name,
                           SLsizei        count,
                           const SLfloat* value,
                           GLboolean      transpose = false) const;
    void  uniformMatrix2fv(SLint          loc,
                           SLsizei        count,
                           const SLfloat* value,
                           GLboolean      transpose = false) const;
    SLint uniformMatrix3fv(const SLchar*  name,
                           SLsizei        count,
                           const SLfloat* value,
                           GLboolean      transpose = false) const;
    void  uniformMatrix3fv(SLint          loc,
                           SLsizei        count,
                           const SLfloat* value,
                           GLboolean      transpose = false) const;
    SLint uniformMatrix4fv(const SLchar*  name,
                           SLsizei        count,
                           const SLfloat* value,
                           GLboolean      transpose = false) const;
    void  uniformMatrix4fv(SLint          loc,
                           SLsizei        count,
                           const SLfloat* value,
                           GLboolean      transpose = false) const;

protected:
    SLuint       _progID;     //!< OpenGL shader program object ID
    SLbool       _isLinked;   //!< Flag if program is linked
    SLVGLShader  _shaders;    //!< Vector of all shader objects
    SLVUniform1f _uniforms1f; //!< Vector of uniform1f variables
    SLVUniform1i _uniforms1i; //!< Vector of uniform1i variables
};
//-----------------------------------------------------------------------------
//! STL vector of SLGLProgram pointers
typedef vector<SLGLProgram*> SLVGLProgram;
//-----------------------------------------------------------------------------
#endif // SLSHADERPROGRAM_H
