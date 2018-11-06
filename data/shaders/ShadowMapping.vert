//#############################################################################
//  File:      ShadowMapping.vert
//  Author:    Micha Stettler
//  Date:      July 2014
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#version 120

uniform mat4 ShadowMapMatrix[8];

varying vec4 ShadowCoord0;
varying vec4 ShadowCoord1;
varying vec4 ShadowCoord2;
varying vec4 ShadowCoord3;
varying vec4 ShadowCoord4;
varying vec4 ShadowCoord5;
varying vec4 ShadowCoord6;
varying vec4 ShadowCoord7;

varying vec3 N_VS;
varying vec3 P_VS;

void main()
{
    //Calculate and Save the Texture Coordinates (Varying Array dont work!)

    if (u_light[0].position.w >= 0.0)
    {	ShadowCoord0 = ShadowMapMatrix[0] * gl_ModelViewMatrix * gl_Vertex;
    }
    if (u_light[1].position.w >= 0.0)
    {	ShadowCoord1 = ShadowMapMatrix[1] * gl_ModelViewMatrix * gl_Vertex;
    }
    if (u_light[2].position.w >= 0.0)
    {	ShadowCoord2 = ShadowMapMatrix[2] * gl_ModelViewMatrix * gl_Vertex;
    }
    if (u_light[3].position.w >= 0.0)
    {	ShadowCoord3 = ShadowMapMatrix[3] * gl_ModelViewMatrix * gl_Vertex;
    }
    if (u_light[4].position.w >= 0.0)
    {	ShadowCoord4 = ShadowMapMatrix[4] * gl_ModelViewMatrix * gl_Vertex;
    }
    if (u_light[5].position.w >= 0.0)
    {	ShadowCoord5 = ShadowMapMatrix[5] * gl_ModelViewMatrix * gl_Vertex;
    }
    if (u_light[6].position.w >= 0.0)
    {	ShadowCoord6 = ShadowMapMatrix[6] * gl_ModelViewMatrix * gl_Vertex;
    }
    if (u_light[7].position.w >= 0.0)
    {	ShadowCoord7 = ShadowMapMatrix[7] * gl_ModelViewMatrix * gl_Vertex;
    }

    // Normal in view space
    N_VS = vec3(gl_NormalMatrix * gl_Normal);

    // Position in view space
    P_VS = vec3(gl_ModelViewMatrix * gl_Vertex);

    gl_Position = ftransform();
}
