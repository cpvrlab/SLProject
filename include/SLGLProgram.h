//#############################################################################
//  File:      SLGLProgram.h
//  Author:    Marcus Hudritsch 
//             Mainly based on Martin Christens GLSL Tutorial
//             See http://www.clockworkcoders.com
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLPROGRAM_H
#define SLGLPROGRAM_H

#include <stdafx.h>
#include "SLGLUniform.h"

class SLGLShader;
class SLScene;
class SLMaterial;

//-----------------------------------------------------------------------------
//! STL vector type for SLGLShader pointers
typedef std::vector<SLGLShader*>  SLVGLShader;

#if defined(TARGET_OS_IOS)
// The TR1 unordered_map or the hash_map is not yet available on iOS
typedef map<string, int> SLLocMap;
#else
//typedef unordered_map<const char*, int, hash<const char*>, eqstr> SLLocMap;
typedef map<string, int> SLLocMap;
#endif

//-----------------------------------------------------------------------------
//! Encapsulation of an OpenGL shader program object
/*!
The SLGLProgram base class represents a shader program object of the OpenGL
Shading Language (GLSL). Multiple SLGLShader objects can be attached and linked 
at run time. An SLGLProgram object can then be attached to an SLMaterial
node for execution. An SLGLProgram object can hold an array of uniform
variable that can transfer variables from the CPU program to the GPU program.
For more details on GLSL please refer to official GLSL documentation.
*/
//-----------------------------------------------------------------------------
class SLGLProgram : public SLObject
{
    public:
                        SLGLProgram     (SLstring vertShaderFile=0,
                                         SLstring fragShaderFile=0);          
    virtual            ~SLGLProgram     ();

            void        addShader       (SLGLShader* shader);         
            void        init            (); //!< create, attach & link shaders
            char*       getLinkerLog    (); //!< get linker messages
      
    virtual void        beginShader     (SLMaterial* mat) = 0;  //!< starter for derived classes
    virtual void        endShader       () = 0;
      
            void        beginUse        (SLMaterial* mat = 0);  //!< begin using shader
            void        endUse          ();
            void        useProgram      ();
      
            void        addUniform1f    (SLGLUniform1f* u);   //!< add float uniform
            void        addUniform1i    (SLGLUniform1i* u);   //!< add int uniform
      
            //Getters
            SLuint      programObjectGL (){return _objectGL;}
            SLVGLShader&  shaders         (){return _shaders;}

      
        //Variable location getters
    inline  SLint       getUniformLocation(const SLchar *name){return glGetUniformLocation(_objectGL, name);}
    inline  SLint       getAttribLocation(const SLchar *name){return glGetAttribLocation(_objectGL, name);}

            //Send unform variables to program
            SLint       uniform1f       (const SLchar* name, SLfloat v0);
            SLint       uniform2f       (const SLchar* name, SLfloat v0, 
                                         SLfloat v1); 
            SLint       uniform3f       (const SLchar* name, SLfloat v0, 
                                         SLfloat v1, SLfloat v2);
            SLint       uniform4f       (const SLchar* name, SLfloat v0, 
                                         SLfloat v1, SLfloat v2, SLfloat v3);

            SLint       uniform1i       (const SLchar* name, SLint v0);
            SLint       uniform2i       (const SLchar* name, SLint v0, 
                                         SLint v1);
            SLint       uniform3i       (const SLchar* name, SLint v0, 
                                         SLint v1, SLint v2);
            SLint       uniform4i       (const SLchar* name, SLint v0, 
                                         SLint v1, SLint v2, SLint v3);

            SLint       uniform1fv      (const SLchar* name, SLsizei count, 
                                         const SLfloat* value);
            SLint       uniform2fv      (const SLchar* name, SLsizei count, 
                                         const SLfloat* value);
            SLint       uniform3fv      (const SLchar* name, SLsizei count, 
                                         const SLfloat* value);
            SLint       uniform4fv      (const SLchar* name, SLsizei count, 
                                         const SLfloat* value);

            SLint       uniform1iv      (const SLchar* name, SLsizei count, 
                                         const SLint* value);
            SLint       uniform2iv      (const SLchar* name, SLsizei count, 
                                         const SLint* value);
            SLint       uniform3iv      (const SLchar* name, SLsizei count, 
                                         const SLint* value);
            SLint       uniform4iv      (const SLchar* name, GLsizei count, 
                                         const SLint* value);

            SLint       uniformMatrix2fv(const SLchar* name, SLsizei count, 
                                         const SLfloat* value, 
                                         GLboolean transpose=false);
            void        uniformMatrix2fv(const SLint loc, SLsizei count, 
                                         const SLfloat* value, 
                                         GLboolean transpose=false);
            SLint       uniformMatrix3fv(const SLchar* name, SLsizei count, 
                                         const SLfloat* value, 
                                         GLboolean transpose=false);
            void        uniformMatrix3fv(const SLint loc, SLsizei count, 
                                         const SLfloat* value, 
                                         GLboolean transpose=false);
            SLint       uniformMatrix4fv(const SLchar* name, SLsizei count, 
                                         const SLfloat* value, 
                                         GLboolean transpose=false); 
            void        uniformMatrix4fv(const SLint loc, SLsizei count, 
                                         const SLfloat* value, 
                                         GLboolean transpose=false); 
      // statics
    static  SLstring   defaultPath;     //!< default path for GLSL programs
      
    private:
        SLGLState*    _stateGL;         //!< Pointer to global SLGLState instance
        SLuint        _objectGL;        //!< OpenGL shader program object
        SLbool        _isLinked;        //!< Flag if program is linked
        SLVGLShader   _shaders;         //!< Vector of all shader objects
        SLVUniform1f  _uniforms1f;      //!< Vector of uniform1f variables
        SLVUniform1i  _uniforms1i;      //!< Vector of uniform1i variables
        //SLLocMap    _uniformLocHash;  //!< Hashmap for all uniform locations
        //SLLocMap    _attribLocHash;   //!< Hashmap for all attribute locations
};
//-----------------------------------------------------------------------------
//! STL vector of SLGLProgram pointers
typedef std::vector<SLGLProgram*> SLVGLProgram;
//-----------------------------------------------------------------------------
#endif // SLSHADERPROGRAM_H

