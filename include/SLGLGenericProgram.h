//#############################################################################
//  File:      SLGLProgramGeneric.h
//  Author:    Marcus Hudritsch
//  Purpose:   Defines a minimal shader program that just starts and stops the
//             shaders that are hold in the base class SLGLProgram.
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLGENERICPROGRAM_H
#define SLGLGENERICPROGRAM_H

#include "SLObject.h"
#include "SLGLProgram.h"

class SLMaterial;

//-----------------------------------------------------------------------------
//! Generic Shader Program class inherited from SLGLProgram
/*!
This class only provides the shader begin and end methods. It can be used for
simple GLSL shader programs with standard types of uniform variables.
*/
class SLGLGenericProgram : public SLGLProgram
{
    public:
                         SLGLGenericProgram(const char* vertShaderFile,
                                            const char* fragShaderFile)
                         : SLGLProgram(vertShaderFile, fragShaderFile) {;}
   
            void        beginShader (SLMaterial* mat) {beginUse(mat);}
            void        endShader   () {endUse();}
};
//-----------------------------------------------------------------------------
#endif
