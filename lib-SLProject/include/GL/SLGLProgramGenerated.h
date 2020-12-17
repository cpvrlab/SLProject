//#############################################################################
//  File:      SLGLProgramGenerated.h
//  Author:    Marcus Hudritsch
//  Purpose:   Defines a generated shader program that just starts and stops the
//             shaders that are hold in the base class SLGLProgram.
//  Date:      December 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLPROGRAMMGENERATED_H
#define SLGLPROGRAMMGENERATED_H

#include <SLGLProgram.h>

class SLMaterial;
class SLAssetManager;

//-----------------------------------------------------------------------------
//! Generated Shader Program class inherited from SLGLProgram
/*!
*/
class SLGLProgramGenerated : public SLGLProgram
{
public:
    ~SLGLProgramGenerated() override = default;

    //! ctor for generated shader program
    SLGLProgramGenerated(SLAssetManager* am,
                         SLMaterial*     mat,
                         SLCamera*       cam,
                         SLVLight*       lights)
      : SLGLProgram(am, "", "")
    {
        buildShaderProgram(mat, cam, lights);
    }

    void buildShaderProgram(SLMaterial* mat,
                            SLCamera*   cam,
                            SLVLight*   lights);
    void beginShader(SLCamera*   cam,
                     SLMaterial* mat,
                     SLVLight*   lights) override { beginUse(cam, mat, lights); }
    void endShader() override { endUse(); }

private:
    // Blinn-Phong shader builder functions
    // Tm = Texture Mapping
    // Nm = Normal Mapping
    // Pm = Parallax Mapping
    // Ao = Ambient Occlusion
    // Sm = Shadow Mapping
    void buildPerPixBlinnTmNmAoSm(SLMaterial* mat, SLCamera* cam, SLVLight* lights);
    void buildPerPixBlinnTmNmAo(SLMaterial* mat, SLCamera* cam, SLVLight* lights);
    void buildPerPixBlinnTmNmSm(SLMaterial* mat, SLCamera* cam, SLVLight* lights);
    void buildPerPixBlinnTmNm(SLMaterial* mat, SLCamera* cam, SLVLight* lights);
    void buildPerPixBlinnTmSm(SLMaterial* mat, SLCamera* cam, SLVLight* lights);
    void buildPerPixBlinnAoSm(SLMaterial* mat, SLCamera* cam, SLVLight* lights);
    void buildPerPixBlinnSm(SLMaterial* mat, SLCamera* cam, SLVLight* lights);
    void buildPerPixBlinnTm(SLMaterial* mat, SLCamera* cam, SLVLight* lights);
    void buildPerPixBlinn(SLMaterial* mat, SLCamera* cam, SLVLight* lights);

    // Helpers
    void   addShadowMapDeclaration(SLVLight* lights, string& fragCode);
    void   addShadowTestCode(SLVLight* lights, string& fragCode);
    string shadowMapUniformName(SLVLight* lights, int lightNum);
};
//-----------------------------------------------------------------------------
#endif
