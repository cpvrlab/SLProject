//#############################################################################
//  File:      SLGLShaderProgGeneric.h
//  Author:    Marcus Hudritsch
//  Purpose:   Defines a minimal shader program that just starts and stops the
//             shaders that are hold in the base class SLGLShaderProg.
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLSHADERPROGGENERIC_H
#define SLSHADERPROGGENERIC_H

#include "SLObject.h"
#include "SLGLShaderProg.h"

class SLMaterial;

//-----------------------------------------------------------------------------
//! Generic Shader Program class inherited from SLGLShaderProg
/*!
This class only provides the shader begin and end methods. It can be used for
simple GLSL shader programs with standard types of uniform variables.
*/
class SLGLShaderProgGeneric : public SLGLShaderProg
{
    public:
                         SLGLShaderProgGeneric(const char* vertShaderFile, 
                                               const char* fragShaderFile)
                         : SLGLShaderProg(vertShaderFile, fragShaderFile) {;}
   
            void        beginShader (SLMaterial* mat) {beginUse(mat);}
            void        endShader   () {endUse();}
};
//-----------------------------------------------------------------------------
#endif