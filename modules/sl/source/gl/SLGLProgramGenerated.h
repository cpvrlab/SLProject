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
/*! An instance of this class generates the shader code on the fly at
 construction time based on the information of the passed material and lights
 vector. The generated program depends on the following parameters:
 - mat->lightModel (only Blinn-Phong is implemented yet)
 - mat->textures:
   - Tm = Texture Mapping with diffuse color map
   - Nm = Normal Mapping with normal map
   - Ao = Ambient Occlusion Mapping with AO map that uses uv2 in SLMesh
 - light->createsShadows
   - Sm = Shadow Map (single or cube shadow map)

 The shader program gets a unique name with the following pattern:
<pre>
 genPerPixBlinnTmNmAo-DsPSs
          |    | | |  ||||+ light before w. shadows
          |    | | |  |||+ Spot light
          |    | | |  ||+ Point light
          |    | | |  |+ light before w. shadows
          |    | | |  + Directional light
          |    | | + Ambient Occlusion
          |    | + Normal Mapping
          |    + Texture Mapping
          + Blinn lighting model
</pre>
 The above example is for a material with 3 textures and a scene with 3
 lights where the first directional light and the third spot light generate
 shadows.
 The shader program is constructed when a material is for the first time
 activated (SLMaterial::activate) and it's program pointer is null. The old
 system of custom written GLSL shader program is still valid.
 At the end of SLMaterial::activate the generated vertex and fragment shader
 get compiled, linked and activated with the OpenGL functions in SLGLShader
 and SLGLProgram.
 After successful compilation the shader get exported into the applications
 config directory if they not yet exist there.
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
      : SLGLProgram(am,
                    "",
                    "",
                    "",
                    programName)
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
    void buildPerPixBlinnSc(SLVLight* lights);
    void buildPerPixBlinnAo(SLVLight* lights);
    void buildPerPixBlinnNm(SLVLight* lights);
    void buildPerPixBlinnTm(SLVLight* lights);
    void buildPerPixBlinn(SLVLight* lights);

    // Helpers

    static string coloredShadows();
    static string fragInputs_u_lightSm(SLVLight* lights);
    static string fragInputs_u_shadowMaps(SLVLight* lights);
    static string fragShadowTest(SLVLight* lights);
    static string shaderHeader(int numLights);
    static void   addCodeToShader(SLGLShader*   shader,
                                  const string& code,
                                  const string& name);
    static string generatedShaderPath; //! Path to write out generated shaders
};
//-----------------------------------------------------------------------------
#endif
