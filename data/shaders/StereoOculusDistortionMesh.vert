//#############################################################################
//  File:      StereoOculusDistortionMesh.vert
//  Purpose:   Oculus Rift Distortion Shader
//  Author:    Marc Wacker
//  Date:      November 2013
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

attribute vec2  a_position;
attribute float a_timeWarpFactor;
attribute float a_vignetteFactor;
attribute vec2  a_texCoordR;
attribute vec2  a_texCoordG;
attribute vec2  a_texCoordB;

uniform   vec2  u_eyeToSourceUVScale;
uniform   vec2  u_eyeToSourceUVOffset;
uniform   mat4  u_eyeRotationStart;
uniform   mat4  u_eyeRotationEnd;

varying   vec2  v_texCoordR;
varying   vec2  v_texCoordG;
varying   vec2  v_texCoordB;
varying   float v_vignette;
varying   float v_timeWarp;

vec2 timewarpTexCoord(vec2 texCoord)
{
    // Vertex inputs are in TanEyeAngle space for the R,G,B channels (i.e. after chromatic
    // aberration and distortion). These are now "real world" vectors in direction (x,y,1)
    // relative to the eye of the HMD. Apply the 3x3 timewarp rotation to these vectors.
    vec3 transformedStart = (u_eyeRotationStart * vec4(texCoord, 1.0, 0.0)).xyz;
    vec3 transformedEnd = (u_eyeRotationEnd * vec4(texCoord, 1.0, 0.0)).xyz;

    vec3 transformed = mix(transformedStart, transformedEnd, a_timeWarpFactor);


    // Project them back onto the Z=1 plane of the rendered images.
    vec2 flattened = (transformed.xy / transformed.z);

    // Scale them into ([0,0.5],[0,1]) or ([0.5,0],[0,1]) UV lookup space (depending on eye)
    return(u_eyeToSourceUVScale * flattened + u_eyeToSourceUVOffset);
}

void main()
{
    v_texCoordR = timewarpTexCoord(a_texCoordR);
    v_texCoordG = timewarpTexCoord(a_texCoordG);
    v_texCoordB = timewarpTexCoord(a_texCoordB);
    v_timeWarp = a_timeWarpFactor;
    v_vignette = a_vignetteFactor;
    gl_Position = vec4(a_position.xy, 0.5, 1.0);
}




/* original hlsl shader below
float2 EyeToSourceUVScale, EyeToSourceUVOffset;
float4x4 EyeRotationStart, EyeRotationEnd;
float2 TimewarpTexCoord(float2 TexCoord, float4x4 rotMat)
{
    // Vertex inputs are in TanEyeAngle space for the R,G,B channels (i.e. after chromatic
    // aberration and distortion). These are now "real world" vectors in direction (x,y,1)
    // relative to the eye of the HMD. Apply the 3x3 timewarp rotation to these vectors.
    float3 transformed = float3( mul ( rotMat, float4(TexCoord.xy, 1, 1) ).xyz);

    // Project them back onto the Z=1 plane of the rendered images.
    float2 flattened = (transformed.xy / transformed.z);

    // Scale them into ([0,0.5],[0,1]) or ([0.5,0],[0,1]) UV lookup space (depending on eye)
    return(EyeToSourceUVScale * flattened + EyeToSourceUVOffset);
}
void main(in float2 Position : POSITION, in float timewarpLerpFactor : POSITION1,
            in float Vignette : POSITION2, in float2 TexCoord0 : TEXCOORD0,
            in float2 TexCoord1 : TEXCOORD1, in float2 TexCoord2 : TEXCOORD2,
            out float4 oPosition : SV_Position, out float2 oTexCoord0 : TEXCOORD0,
            out float2 oTexCoord1 : TEXCOORD1, out float2 oTexCoord2 : TEXCOORD2,
            out float oVignette : TEXCOORD3)
{
    float4x4 lerpedEyeRot = lerp(EyeRotationStart, EyeRotationEnd, timewarpLerpFactor);
    oTexCoord0 = TimewarpTexCoord(TexCoord0,lerpedEyeRot);
    oTexCoord1 = TimewarpTexCoord(TexCoord1,lerpedEyeRot);
    oTexCoord2 = TimewarpTexCoord(TexCoord2,lerpedEyeRot);
    oPosition = float4(Position.xy, 0.5, 1.0);
    oVignette = Vignette; // For vignette fade
}

*/