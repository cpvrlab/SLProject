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
                         const string&   programName,
                         SLMaterial*     mat,
                         SLVLight*       lights)
      : SLGLProgram(am, "", "", "", programName)
    {
        buildProgramCode(mat, lights);
    }

    static bool lightsDoShadowMapping(SLVLight* lights);
    static void buildProgramName(SLMaterial* mat,
                                 SLVLight*   lights,
                                 string&     programName);
    void        buildProgramCode(SLMaterial* mat,
                                 SLVLight*   lights);
    void        beginShader(SLCamera*   cam,
                            SLMaterial* mat,
                            SLVLight*   lights) override { beginUse(cam, mat, lights); }
    void        endShader() override { endUse(); }

private:
    // Blinn-Phong shader builder functions
    // Tm = Texture Mapping
    // Nm = Normal Mapping
    // Pm = Parallax Mapping
    // Ao = Ambient Occlusion
    // Sm = Shadow Mapping
    void buildPerPixBlinnTmNmAoSm(SLVLight* lights);
    void buildPerPixBlinnTmNmAo(SLVLight* lights);
    void buildPerPixBlinnTmNmSm(SLVLight* lights);
    void buildPerPixBlinnTmAoSm(SLVLight* lights);
    void buildPerPixBlinnAoSm(SLVLight* lights);
    void buildPerPixBlinnNmSm(SLVLight* lights);
    void buildPerPixBlinnTmSm(SLVLight* lights);
    void buildPerPixBlinnNmAo(SLVLight* lights);
    void buildPerPixBlinnTmAo(SLVLight* lights);
    void buildPerPixBlinnTmNm(SLVLight* lights);
    void buildPerPixBlinnSm(SLVLight* lights);
    void buildPerPixBlinnAo(SLVLight* lights);
    void buildPerPixBlinnNm(SLVLight* lights);
    void buildPerPixBlinnTm(SLVLight* lights);
    void buildPerPixBlinn(SLVLight* lights);

    // Helpers
    string fragInputs_u_shadowMaps(SLVLight* lights);
    string fragShadowTest(SLVLight* lights);
    string shaderHeader(int numLights);
    void   addCodeToShader(SLGLShader*   shader,
                           const string& code,
                           const string& name);
};
//-----------------------------------------------------------------------------
#endif
