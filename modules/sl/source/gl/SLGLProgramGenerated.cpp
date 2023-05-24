//#############################################################################
//  File:      SLGLProgramGenerated.cpp
//  Date:      December 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLAssetManager.h>
#include <SLGLProgramManager.h>
#include <SLGLProgramGenerated.h>
#include <SLGLShader.h>
#include <SLCamera.h>
#include <SLLight.h>
#include <SLGLDepthBuffer.h>

using std::string;
using std::to_string;

///////////////////////////////
// Const. GLSL code snippets //
///////////////////////////////

//-----------------------------------------------------------------------------
string SLGLProgramGenerated::generatedShaderPath;
//-----------------------------------------------------------------------------
const string vertInput_a_pn               = R"(
layout (location = 0) in vec4  a_position;       // Vertex position attribute
layout (location = 1) in vec3  a_normal;         // Vertex normal attribute)";
const string vertInput_PS_a_p             = R"(
layout (location = 0) in vec3  a_position;       // Particle position attribute)";
const string vertInput_PS_a_v             = R"(
layout (location = 1) in vec3  a_velocity;       // Particle velocity attribute)";
const string vertInput_PS_a_st            = R"(
layout (location = 2) in float a_startTime;      // Particle start time attribute)";
const string vertInput_PS_a_initV         = R"(
layout (location = 3) in vec3  a_initialVelocity;// Particle initial velocity attribute)";
const string vertInput_PS_a_r             = R"(
layout (location = 4) in float a_rotation;       // Particle rotation attribute)";
const string vertInput_PS_a_r_angularVelo = R"(
layout (location = 5) in float a_angularVelo;    // Particle rotation rate attribute)";
const string vertInput_PS_a_texNum        = R"(
layout (location = 6) in uint  a_texNum;         // Particle rotation attribute)";
const string vertInput_PS_a_initP         = R"(
layout (location = 7) in vec3  a_initialPosition;// Particle initial position attribute)";
const string vertInput_a_uv0              = R"(
layout (location = 2) in vec2  a_uv0;            // Vertex tex.coord. 1 for diffuse color)";
const string vertInput_a_uv1              = R"(
layout (location = 3) in vec2  a_uv1;            // Vertex tex.coord. 2 for AO)";
const string vertInput_a_tangent          = R"(
layout (location = 5) in vec4  a_tangent;        // Vertex tangent attribute)";
//-----------------------------------------------------------------------------
const string vertInput_u_matrices_all = R"(

uniform mat4  u_mMatrix;    // Model matrix (object to world transform)
uniform mat4  u_vMatrix;    // View matrix (world to camera transform)
uniform mat4  u_pMatrix;    // Projection matrix (camera to normalize device coords.))";
const string vertInput_u_matrix_vOmv  = R"(
uniform mat4  u_vOmvMatrix;         // view or modelview matrix)";
//-----------------------------------------------------------------------------
const string vertInput_u_lightNm = R"(

uniform vec4  u_lightPosVS[NUM_LIGHTS];     // position of light in view space
uniform vec3  u_lightSpotDir[NUM_LIGHTS];   // spot direction in view space
uniform float u_lightSpotDeg[NUM_LIGHTS];   // spot cutoff angle 1-180 degrees)";
//-----------------------------------------------------------------------------
const string vertConstant_PS_pi = R"(

#define PI 3.1415926538
#define TWOPI 6.2831853076
)";
//-----------------------------------------------------------------------------
const string vertInput_PS_u_time               = R"(

uniform float u_time;               // Simulation time
uniform float u_difTime;            // Simulation delta time after frustum culling
uniform float u_tTL;                // Time to live of a particle)";
const string vertInput_PS_u_al_bernstein_alpha = R"(
uniform vec4  u_al_bernstein;       // Bernstein polynomial for alpha over time)";
const string vertInput_PS_u_al_bernstein_size  = R"(
uniform vec4  u_si_bernstein;       // Bernstein polynomial for size over time)";
const string vertInput_PS_u_colorOvLF          = R"(
uniform float u_colorArr[256 * 3];  // Array of color value (for color over life))";
const string vertInput_PS_u_deltaTime          = R"(
uniform float u_deltaTime;          // Elapsed time between frames)";
const string vertInput_PS_u_pgPos              = R"(
uniform vec3  u_pGPosition;         // Particle Generator position)";
const string vertInput_PS_u_a_const            = R"(
uniform float u_accConst;           // Particle acceleration constant)";
const string vertInput_PS_u_a_diffDir          = R"(
uniform vec3  u_acceleration;       // Particle acceleration)";
const string vertInput_PS_u_g                  = R"(
uniform vec3  u_gravity;            // Particle gravity)";
const string vertInput_PS_u_angularVelo        = R"(
uniform float u_angularVelo;        // Particle angular velocity)";
const string vertInput_PS_u_col                = R"(
uniform int   u_col;                // Number of column of flipbook texture)";
const string vertInput_PS_u_row                = R"(
uniform int   u_row;                // Number of row of flipbook texture)";
const string vertInput_PS_u_condFB             = R"(
uniform int   u_condFB;             // Condition to update texNum)";

//-----------------------------------------------------------------------------
const string vertOutput_v_P_VS       = R"(

out     vec3  v_P_VS;                   // Point of illumination in view space (VS))";
const string vertOutput_v_P_WS       = R"(
out     vec3  v_P_WS;                   // Point of illumination in world space (WS))";
const string vertOutput_v_N_VS       = R"(
out     vec3  v_N_VS;                   // Normal at P_VS in view space (VS))";
const string vertOutput_v_R_OS       = R"(
out     vec3  v_R_OS;                   // Reflection vector in object space (WS))";
const string vertOutput_v_uv0        = R"(
out     vec2  v_uv0;                    // Texture coordinate 1 output)";
const string vertOutput_v_uv1        = R"(
out     vec2  v_uv1;                    // Texture coordinate 1 output)";
const string vertOutput_v_lightVecTS = R"(
out     vec3  v_eyeDirTS;               // Vector to the eye in tangent space
out     vec3  v_lightDirTS[NUM_LIGHTS]; // Vector to the light 0 in tangent space
out     vec3  v_spotDirTS[NUM_LIGHTS];  // Spot direction in tangent space)";
//-----------------------------------------------------------------------------
const string vertFunction_PS_ColorOverLT = R"(

vec3 colorByAge(float age)
{
    int  cachePos = int(clamp(age, 0.0, 1.0) * 255.0) * 3;
    vec3 color    = vec3(u_colorArr[cachePos], u_colorArr[cachePos + 1], u_colorArr[cachePos + 2]);
    return color;
})";

//-----------------------------------------------------------------------------
const string vertOutput_PS_struct_Begin  = R"(

out vertex
{ )";
const string vertOutput_PS_struct_t      = R"(
    float transparency; // Transparency of a particle)";
const string vertOutput_PS_struct_r      = R"(
    float rotation;     // Rotation of a particle)";
const string vertOutput_PS_struct_s      = R"(
    float size;         // Size of a particle )";
const string vertOutput_PS_struct_c      = R"(
    vec3 color;         // Color of a particle )";
const string vertOutput_PS_struct_texNum = R"(
    uint texNum;        // Num of texture in flipbook)";
const string vertOutput_PS_struct_End    = R"(
} vert; )";
//-----------------------------------------------------------------------------
const string vertOutput_PS_tf_p             = R"(

out     vec3  tf_position;          // To transform feedback)";
const string vertOutput_PS_tf_v             = R"(
out     vec3  tf_velocity;          // To transform feedback)";
const string vertOutput_PS_tf_st            = R"(
out     float tf_startTime;         // To transform feedback)";
const string vertOutput_PS_tf_initV         = R"(
out     vec3  tf_initialVelocity;   // To transform feedback)";
const string vertOutput_PS_tf_r             = R"(
out     float tf_rotation;          // To transform feedback)";
const string vertOutput_PS_tf_r_angularVelo = R"(
out     float tf_angularVelo;       // To transform feedback)";
const string vertOutput_PS_tf_texNum        = R"(
out     uint  tf_texNum;            // To transform feedback)";
const string vertOutput_PS_tf_initP         = R"(
out     vec3  tf_initialPosition;   // To transform feedback)";

//-----------------------------------------------------------------------------
const string vertMain_Begin              = R"(

void main()
{)";
const string vertMain_v_P_VS             = R"(
    mat4 mvMatrix = u_vMatrix * u_mMatrix;
    v_P_VS = vec3(mvMatrix *  a_position);   // vertex position in view space)";
const string vertMain_v_P_WS_Sm          = R"(
    v_P_WS = vec3(u_mMatrix * a_position);   // vertex position in world space)";
const string vertMain_v_N_VS             = R"(
    mat3 invMvMatrix = mat3(inverse(mvMatrix));
    mat3 nMatrix = transpose(invMvMatrix);
    v_N_VS = vec3(nMatrix * a_normal);       // vertex normal in view space)";
const string vertMain_v_R_OS             = R"(
    vec3 I = normalize(v_P_VS);
    vec3 N = normalize(v_N_VS);
    v_R_OS = invMvMatrix * reflect(I, N); // R = I-2.0*dot(N,I)*N;)";
const string vertMain_v_uv0              = R"(

    v_uv0 = a_uv0;  // pass diffuse color tex.coord. 1 for interpolation)";
const string vertMain_v_uv1              = R"(
    v_uv1 = a_uv1;  // pass diffuse color tex.coord. 1 for interpolation)";
const string vertMain_TBN_Nm             = R"(

    // Building the matrix Eye Space -> Tangent Space
    // See the math behind at: http://www.terathon.com/code/tangent.html
    vec3 n = normalize(nMatrix * a_normal);
    vec3 t = normalize(nMatrix * a_tangent.xyz);
    vec3 b = cross(n, t) * a_tangent.w; // bitangent w. corrected handedness
    mat3 TBN = mat3(t,b,n);

    // Transform vector to the eye into tangent space
    v_eyeDirTS = -v_P_VS;  // eye vector in view space
    v_eyeDirTS *= TBN;

    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        // Transform spot direction into tangent space
        v_spotDirTS[i] = u_lightSpotDir[i];
        v_spotDirTS[i]  *= TBN;

        // Transform vector to the light 0 into tangent space
        vec3 L = u_lightPosVS[i].xyz - v_P_VS;
        v_lightDirTS[i] = L;
        v_lightDirTS[i] *= TBN;
    }
)";
const string vertMain_PS_v_a             = R"(
    float age = u_time - a_startTime;   // Get the age of the particle)";
const string vertMain_PS_v_t_default     = R"(
    if(age < 0.0)
        vert.transparency = 0.0; // To be discard, because the particle is to be born
    else
        vert.transparency = 1.0;)";
const string vertMain_PS_v_t_begin       = R"(
    if(age < 0.0)
        vert.transparency = 0.0; // To be discard, because the particle is to be born
    else
    {
        vert.transparency = age / u_tTL;  // Get by the ratio age:lifetime)";
const string vertMain_PS_v_t_linear      = R"(
        vert.transparency = 1.0 - vert.transparency;  // Linear)";
const string vertMain_PS_v_t_curve       = R"(
        vert.transparency = pow(vert.transparency,3.0) * u_al_bernstein.x +
                            pow(vert.transparency,2.0) * u_al_bernstein.y +
                            vert.transparency * u_al_bernstein.z +
                            u_al_bernstein.w;  // Get transparency by bezier curve)";
const string vertMain_PS_v_t_end         = R"(
    })";
const string vertMain_PS_v_r             = R"(
    vert.rotation = a_rotation;)";
const string vertMain_PS_v_s             = R"(
    vert.size = age / u_tTL;)";
const string vertMain_PS_v_s_curve       = R"(
    vert.size = pow(vert.size,3.0) * u_si_bernstein.x +
                pow(vert.size,2.0) * u_si_bernstein.y +
                vert.size * u_si_bernstein.z +
                u_si_bernstein.w;  // Get transparency by bezier curve)";
const string vertMain_PS_v_doColorOverLT = R"(
    vert.color = colorByAge(age/u_tTL);)";
const string vertMain_PS_v_texNum        = R"(
    vert.texNum = a_texNum;)";
const string vertMain_PS_EndAll          = R"(

    // Modelview matrix multiplication with (particle position + particle generator position)
    // Calculate position in view space
    gl_Position =  u_vOmvMatrix * vec4(a_position, 1);
}
)";

const string vertMain_PS_EndAll_VertBillboard = R"(
    gl_Position =  vec4(a_position, 1);
}
)";

const string vertMain_EndAll = R"(

    // pass the vertex w. the fix-function transform
    gl_Position = u_pMatrix * mvMatrix * a_position;
}
)";
//-----------------------------------------------------------------------------
const string vertMain_PS_U_Begin                = R"(

void main()
{
    vec4 P = vec4(a_position.xyz, 1.0); // Need to be here for the compilation
    gl_Position = P;                    // Need to be here for the compilation)";
const string vertMain_PS_U_v_init_p             = R"(
    tf_position = a_position;           // Init the output variable)";
const string vertMain_PS_U_v_init_v             = R"(
    tf_velocity = a_velocity;           // Init the output variable)";
const string vertMain_PS_U_v_init_st            = R"(
    tf_startTime = a_startTime;         // Init the output variable)";
const string vertMain_PS_U_v_init_initV         = R"(
    tf_initialVelocity = a_initialVelocity; // Init the output variable)";
const string vertMain_PS_U_v_init_r             = R"(
    tf_rotation = a_rotation;           // Init the output variable)";
const string vertMain_PS_U_v_init_r_angularVelo = R"(
    tf_angularVelo = a_angularVelo;     // Init the output variable)";
const string vertMain_PS_U_v_init_texNum        = R"(
    tf_texNum = a_texNum;               // Init the output variable)";
const string vertMain_PS_U_v_init_initP         = R"(
    tf_initialPosition = a_initialPosition; // Init the output variable)";

const string vertMain_PS_U_bornDead            = R"(
    tf_startTime += u_difTime;          // Add time to resume after frustum culling

    if( u_time >= tf_startTime )
    {   // Check if the particle is born
        float age = u_time - tf_startTime;   // Get the age of the particle
        if( age > u_tTL)
        {   )";
const string vertMain_PS_U_reset_p             = R"(
            // The particle is past its lifetime, recycle.
            tf_position = u_pGPosition; // Reset position)";
const string vertMain_PS_U_reset_shape_p       = R"(
            // The particle is past its lifetime, recycle.
            tf_position = a_initialPosition + u_pGPosition; // Reset position)";
const string vertMain_PS_U_reset_v             = R"(
            tf_velocity = a_initialVelocity; // Reset velocity)";
const string vertMain_PS_U_reset_st_counterGap = R"(
            tf_startTime = u_time + (age - u_tTL); // Reset start time to actual time with counter gap)";
const string vertMain_PS_U_reset_st            = R"(
            tf_startTime = u_time; // Reset start time to actual time)";
const string vertMain_PS_U_alive_p             = R"(
        } else
        {
            // The particle is alive, update.
            tf_position += tf_velocity * u_deltaTime;   // Scale the translation by the delta time)";
const string vertMain_PS_U_v_rConst            = R"(
            tf_rotation = mod(tf_rotation + (u_angularVelo*u_deltaTime), TWOPI);)";
const string vertMain_PS_U_v_rRange            = R"(
            tf_rotation = mod(tf_rotation + (tf_angularVelo*u_deltaTime), TWOPI);)";
const string vertMain_PS_U_alive_a_const       = R"(
            tf_velocity += tf_initialVelocity * u_deltaTime * u_accConst;  // Amplify the velocity)";
const string vertMain_PS_U_alive_a_diffDir     = R"(
            tf_velocity += u_deltaTime * u_acceleration;    // Amplify the velocity)";
const string vertMain_PS_U_alive_g             = R"(
            tf_velocity += u_deltaTime * u_gravity;         // Apply gravity)";
const string vertMain_PS_U_alive_texNum        = R"(
            if(u_condFB == 1)
            {
                tf_texNum++;  // Increment to draw next texture (flipbook)
                tf_texNum = uint(mod(float(tf_texNum), float(u_col * u_row))); // Modulo to not exceed the max and reset
            })";
const string vertMain_PS_U_EndAll              = R"(
        }
    }
})";
//-----------------------------------------------------------------------------
const string geomConfig_PS = R"(
layout (points) in;             // Primitives that we received from vertex shader
layout (triangle_strip, max_vertices = 4) out;    // Primitives that we will output and number of vertex that will be output)";

const string geomInput_PS_struct_Begin  = R"(
in vertex { )";
const string geomInput_PS_struct_t      = R"(
    float transparency; // Transparency of a particle)";
const string geomInput_PS_struct_r      = R"(
    float rotation;     // Rotation of a particle)";
const string geomInput_PS_struct_s      = R"(
    float size;         // Size of a particle )";
const string geomInput_PS_struct_c      = R"(
    vec3 color;         // Color of a particle )";
const string geomInput_PS_struct_texNum = R"(
    uint texNum;        // Num of texture in flipbook)";
const string geomInput_PS_struct_End    = R"(
} vert[]; )";
//-----------------------------------------------------------------------------
const string geomInput_u_matrix_p             = R"(
uniform mat4  u_pMatrix;     // Projection matrix)";
const string geomInput_u_matrix_vertBillboard = R"(
uniform mat4  u_vYawPMatrix; // Projection matrix)";
const string geomInput_PS_u_ScaRa             = R"(

uniform float u_scale;       // Particle scale
uniform float u_radiusW;     // Particle width radius)
uniform float u_radiusH;     // Particle height radius)";
const string geomInput_PS_u_c                 = R"(
uniform vec4  u_color;       // Particle color)";
const string geomInput_PS_u_col               = R"(
uniform int   u_col;         // Number of column of flipbook texture)";
const string geomInput_PS_u_row               = R"(
uniform int   u_row;         // Number of row of flipbook texture)";
//-----------------------------------------------------------------------------
const string geomOutput_PS_v_pC = R"(

out vec4 v_particleColor;   // The resulting color per vertex)";
const string geomOutput_PS_v_tC = R"(
out vec2 v_texCoord;        // Texture coordinate at vertex)";
//-----------------------------------------------------------------------------
const string geomMain_PS_Begin = R"(

void main()
{)";

const string geomMain_PS_v_s             = R"(
    float scale = u_scale;)";
const string geomMain_PS_v_sS            = R"(
    scale *= vert[0].size;)";
const string geomMain_PS_v_rad           = R"(
    float radiusW = u_radiusW * scale;
    float radiusH = u_radiusH * scale;)";
const string geomMain_PS_v_p             = R"(
    vec4 P = gl_in[0].gl_Position;    // Position of the point that we received)";
const string geomMain_PS_v_rot           = R"(
    mat2 rot = mat2(cos(vert[0].rotation),-sin(vert[0].rotation),
                    sin(vert[0].rotation), cos(vert[0].rotation)); // Matrix of rotation)";
const string geomMain_PS_v_rotIden       = R"(
    mat2 rot = mat2(1.0, 0.0, 0.0, 1.0);     // Matrix of rotation)";
const string geomMain_PS_v_c             = R"(
    vec4 color = u_color;                    // Particle color)";
const string geomMain_PS_v_doColorOverLT = R"(
    vec4 color = vec4(vert[0].color, 1.0);   // Particle color)";
const string geomMain_PS_v_withoutColor  = R"(
    vec4 color = vec4( 0.0, 0.0, 0.0, 1.0);  // Particle color)";
const string geomMain_PS_v_cT            = R"(
    color.w *= vert[0].transparency;         // Apply transparency)";

const string geomMain_PS_fourCorners                         = R"(
    //BOTTOM LEFT
    vec4 va = vec4(P.xy + (rot * vec2(-radiusW, -radiusH)), P.z, 1); //Position in view space
    gl_Position = u_pMatrix * va; // Calculate position in clip space
    v_texCoord = vec2(0.0, 0.0);  // Texture coordinate
    v_particleColor = color;
    EmitVertex();

    //BOTTOM RIGHT
    vec4 vd = vec4(P.xy + (rot * vec2(radiusW, -radiusH)), P.z,1);
    gl_Position = u_pMatrix * vd;
    v_texCoord = vec2(1.0, 0.0);
    v_particleColor = color;
    EmitVertex();

    //TOP LEFT
    vec4 vb = vec4(P.xy + (rot * vec2(-radiusW,radiusH)) , P.z,1);
    gl_Position = u_pMatrix * vb;
    v_texCoord = vec2(0.0, 1.0);
    v_particleColor = color;
    EmitVertex();

    //TOP RIGHT
    vec4 vc = vec4(P.xy + (rot *vec2(radiusW, radiusH)), P.z,1);
    gl_Position = u_pMatrix *  vc;
    v_texCoord = vec2(1.0, 1.0);
    v_particleColor = color;
    EmitVertex();)";
const string geomMain_PS_fourCorners_vertBillboard           = R"(

    //BOTTOM LEFT
    vec4 va = vec4(P.xy + (rot * vec2(-radiusW, -radiusH)), P.z, 1); //Position in view space
    gl_Position = u_pMatrix * (u_vYawPMatrix * va); // Calculate position in clip space
    v_texCoord = vec2(0.0, 0.0);  // Texture coordinate
    v_particleColor = color;
    EmitVertex();

    //BOTTOM RIGHT
    vec4 vd = vec4(P.xy + (rot * vec2(radiusW, -radiusH)), P.z,1);
    gl_Position = u_pMatrix * (u_vYawPMatrix  *vd);
    v_texCoord = vec2(1.0, 0.0);
    v_particleColor = color;
    EmitVertex();

    //TOP LEFT
    vec4 vb = vec4(P.xy + (rot * vec2(-radiusW,radiusH)) , P.z,1);
    gl_Position = u_pMatrix * (u_vYawPMatrix * vb);
    v_texCoord = vec2(0.0, 1.0);
    v_particleColor = color;
    EmitVertex();

    //TOP RIGHT
    vec4 vc = vec4(P.xy + (rot *vec2(radiusW, radiusH)), P.z,1);
    gl_Position = u_pMatrix *  (u_vYawPMatrix * vc);
    v_texCoord = vec2(1.0, 1.0);
    v_particleColor = color;
    EmitVertex();  )";
const string geomMain_PS_fourCorners_horizBillboard          = R"(

    //FRONT LEFT
    vec4 va = vec4(P.xyz, 1); //Position in view space
    va.xz = va.xz + (rot * vec2(-radiusW, -radiusH));
    gl_Position = u_pMatrix * u_vOmvMatrix * va; // Calculate position in clip space
    v_texCoord = vec2(0.0, 0.0);  // Texture coordinate
    v_particleColor = color;
    EmitVertex();

    //FRONT RIGHT
    vec4 vd = vec4(P.xyz,1);
    vd.xz += (rot * vec2(radiusW, -radiusH));
    gl_Position = u_pMatrix * u_vOmvMatrix * vd;
    v_texCoord = vec2(1.0, 0.0);
    v_particleColor = color;
    EmitVertex();

    //BACK LEFT
    vec4 vb = vec4(P.xyz,1);
    vb.xz += (rot * vec2(-radiusW,radiusH));
    gl_Position = u_pMatrix * u_vOmvMatrix * vb;
    v_texCoord = vec2(0.0, 1.0);
    v_particleColor = color;
    EmitVertex();

    //BACK RIGHT
    vec4 vc = vec4(P.xyz,1);
    vc.xz += (rot * vec2(radiusW, radiusH));
    gl_Position = u_pMatrix * u_vOmvMatrix * vc;
    v_texCoord = vec2(1.0, 1.0);
    v_particleColor = color;
    EmitVertex();  )";
const string geomMain_PS_Flipbook_fourCorners                = R"(
    uint actCI = uint(mod(float(uint(vert[0].texNum), float(u_col)));
    uint actRI = (vert[0].texNum - actCI) / u_col;
    float actC = float(actCI);
    float actR = float(actRI);

    //BOTTOM LEFT
    vec4 va = vec4(P.xy + (rot * vec2(-radiusW, -radiusH)), P.z, 1); //Position in view space
    gl_Position = u_pMatrix * va; // Calculate position in clip space
    v_texCoord = vec2(actC/u_col, 1.0-((actR+1.0)/u_row));  // Texture coordinate
    v_particleColor = color;
    EmitVertex();

    //BOTTOM RIGHT
    vec4 vd = vec4(P.xy + (rot * vec2(radiusW, -radiusH)), P.z,1);
    gl_Position = u_pMatrix * vd;
    v_texCoord = vec2((actC+1.0)/u_col, 1.0-((actR+1.0)/u_row)); // Texture coordinate
    v_particleColor = color;
    EmitVertex();

    //TOP LEFT
    vec4 vb = vec4(P.xy + (rot * vec2(-radiusW,radiusH)) , P.z,1);
    gl_Position = u_pMatrix * vb;
    v_texCoord = vec2(actC/u_col, 1.0-(actR/u_row)); // Texture coordinate
    v_particleColor = color;
    EmitVertex();

    //TOP RIGHT
    vec4 vc = vec4(P.xy + (rot *vec2(radiusW, radiusH)), P.z,1);
    gl_Position = u_pMatrix *  vc;
    v_texCoord = vec2((actC+1.0)/u_col, 1.0-(actR/u_row)); // Texture coordinate
    v_particleColor = color;
    EmitVertex();  )";
const string geomMain_PS_Flipbook_fourCorners_horizBillboard = R"(
    uint actCI = uint(mod(float(vert[0].texNum), float(u_col)));
    uint actRI = uint((float(vert[0].texNum) - float(actCI)) / float(u_col));
    float actC = float(actCI);
    float actR = float(actRI);

    //FRONT LEFT
    vec4 va = vec4(P.xyz, 1); //Position in view space
    va.xz = va.xz + (rot * vec2(-radiusW, -radiusH));
    gl_Position =  u_pMatrix * u_vOmvMatrix * va; // Calculate position in clip space
    v_texCoord = vec2(actC/u_col, 1.0-((actR+1.0)/u_row));  // Texture coordinate
    v_particleColor = color;
    EmitVertex();

    //FRONT RIGHT
    vec4 vd = vec4(P.xyz,1);
    vd.xz += (rot * vec2(radiusW, -radiusH));
    gl_Position =  u_pMatrix * u_vOmvMatrix * vd;
    v_texCoord = vec2((actC+1.0)/u_col, 1.0-((actR+1.0)/u_row)); // Texture coordinate
    v_particleColor = color;
    EmitVertex();

    //BACK LEFT
    vec4 vb = vec4(P.xyz,1);
    vb.xz += (rot * vec2(-radiusW,radiusH));
    gl_Position =  u_pMatrix * u_vOmvMatrix * vb;
    v_texCoord = vec2(actC/u_col, 1.0-(actR/u_row)); // Texture coordinate
    v_particleColor = color;
    EmitVertex();

    //BACK RIGHT
    vec4 vc = vec4(P.xyz,1);
    vc.xz += (rot * vec2(radiusW, radiusH));
    gl_Position = u_pMatrix * u_vOmvMatrix * vc;
    v_texCoord = vec2((actC+1.0)/u_col, 1.0-(actR/u_row)); // Texture coordinate
    v_particleColor = color;
    EmitVertex();  )";
const string geomMain_PS_Flipbook_fourCorners_vertBillboard  = R"(
    uint actCI = uint(mod(float(vert[0].texNum), float(u_col)));
    uint actRI = uint((float(vert[0].texNum) - float(actCI)) / float(u_col));
    float actC = float(actCI);
    float actR = float(actRI);

    //BOTTOM LEFT
    vec4 va = vec4(P.xy + (rot * vec2(-radiusW, -radiusH)), P.z, 1); //Position in view space
    gl_Position = u_pMatrix * (u_vYawPMatrix * va); // Calculate position in clip space
    v_texCoord = vec2(actC/float(u_col), 1.0-((actR+1.0)/float(u_row)));  // Texture coordinate
    v_particleColor = color;
    EmitVertex();

    //BOTTOM RIGHT
    vec4 vd = vec4(P.xy + (rot * vec2(radiusW, -radiusH)), P.z,1);
    gl_Position = u_pMatrix * (u_vYawPMatrix * vd);
    v_texCoord = vec2((actC+1.0)/float(u_col), 1.0-((actR+1.0)/float(u_row))); // Texture coordinate
    v_particleColor = color;
    EmitVertex();

    //TOP LEFT
    vec4 vb = vec4(P.xy + (rot * vec2(-radiusW,radiusH)) , P.z,1);
    gl_Position = u_pMatrix * (u_vYawPMatrix * vb);
    v_texCoord = vec2(actC/float(u_col), 1.0-(actR/float(u_row))); // Texture coordinate
    v_particleColor = color;
    EmitVertex();

    //TOP RIGHT
    vec4 vc = vec4(P.xy + (rot *vec2(radiusW, radiusH)), P.z,1);
    gl_Position =u_pMatrix * (u_vYawPMatrix * vc);
    v_texCoord = vec2((actC+1.0)/float(u_col), 1.0-(actR/float(u_row))); // Texture coordinate
    v_particleColor = color;
    EmitVertex();  )";

const string geomMain_PS_EndAll = R"(

    EndPrimitive();  // Send primitives to fragment shader
}   )";
//-----------------------------------------------------------------------------
const string fragInput_v_P_VS       = R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS))";
const string fragInput_v_P_WS       = R"(
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS))";
const string fragInput_v_N_VS       = R"(
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space)";
const string fragInput_v_R_OS       = R"(
in      vec3        v_R_OS;     // Interpol. reflect in object space)";
const string fragInput_v_uv0        = R"(
in      vec2        v_uv0;      // Texture coordinate varying)";
const string fragInput_v_uv1        = R"(
in      vec2        v_uv1;      // Texture coordinate varying)";
const string fragInput_v_lightVecTS = R"(
in      vec3        v_eyeDirTS;                 // Vector to the eye in tangent space
in      vec3        v_lightDirTS[NUM_LIGHTS];   // Vector to light 0 in tangent space
in      vec3        v_spotDirTS[NUM_LIGHTS];    // Spot direction in tangent space)";
//-----------------------------------------------------------------------------
const string fragInput_PS_v_pC = R"(
in      vec4        v_particleColor;        // interpolated color from the geometry shader)";
const string fragInput_PS_v_tC = R"(
in      vec2        v_texCoord;             // interpolated texture coordinate)";
//-----------------------------------------------------------------------------
const string fragInput_PS_u_overG     = R"(
uniform float       u_oneOverGamma;         // 1.0f / Gamma correction value)";
const string fragInput_PS_u_wireFrame = R"(
uniform bool        u_doWireFrame;          // Boolean for wireFrame)";
//-----------------------------------------------------------------------------
const string fragMain_PS_TF = R"(

out     vec4     o_fragColor;               // output fragment color

void main()
{     
   o_fragColor = vec4(0,0,0,0); // Need to be here for the compilation
}
)";
//-----------------------------------------------------------------------------
const string fragMain_PS              = R"(

void main()
{     
    // Just set the interpolated color from the vertex shader
   o_fragColor = v_particleColor;

   // componentwise multiply w. texture color
    if(!u_doWireFrame)
        o_fragColor *= texture(u_matTextureDiffuse0, v_texCoord);

   if(o_fragColor.a < 0.001)
        discard;

)";
const string fragMain_PS_withoutColor = R"(
void main()
{     
   // componentwise multiply w. texture color
   if(!u_doWireFrame)
        o_fragColor = texture(u_matTextureDiffuse0, v_texCoord);
   else
        o_fragColor = vec4(0,0,0,1.0);

   o_fragColor.a *= v_particleColor.a;

   if(o_fragColor.a < 0.001)
        discard;

)";
const string fragMain_PS_endAll       = R"(
    //Same color for each wireframe
    if(u_doWireFrame)
        o_fragColor =  vec4(0,0,0,1.0);

   o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));
})";
//-----------------------------------------------------------------------------
const string fragInput_u_lightAll    = R"(

uniform bool        u_lightIsOn[NUM_LIGHTS];        // flag if light is on
uniform vec4        u_lightPosVS[NUM_LIGHTS];       // position of light in view space
uniform vec4        u_lightAmbi[NUM_LIGHTS];        // ambient light intensity (Ia)
uniform vec4        u_lightDiff[NUM_LIGHTS];        // diffuse light intensity (Id)
uniform vec4        u_lightSpec[NUM_LIGHTS];        // specular light intensity (Is)
uniform vec3        u_lightSpotDir[NUM_LIGHTS];     // spot direction in view space
uniform float       u_lightSpotDeg[NUM_LIGHTS];     // spot cutoff angle 1-180 degrees
uniform float       u_lightSpotCos[NUM_LIGHTS];     // cosine of spot cutoff angle
uniform float       u_lightSpotExp[NUM_LIGHTS];     // spot exponent
uniform vec3        u_lightAtt[NUM_LIGHTS];         // attenuation (const,linear,quadr.)
uniform bool        u_lightDoAtt[NUM_LIGHTS];       // flag if att. must be calc.
uniform vec4        u_globalAmbi;                   // Global ambient scene color
uniform float       u_oneOverGamma;                 // 1.0f / Gamma correction value
)";
const string fragInput_u_matBlinnAll = R"(
uniform vec4        u_matAmbi;                      // ambient color reflection coefficient (ka)
uniform vec4        u_matDiff;                      // diffuse color reflection coefficient (kd)
uniform vec4        u_matSpec;                      // specular color reflection coefficient (ks)
uniform vec4        u_matEmis;                      // emissive color for self-shining materials
uniform float       u_matShin;                      // shininess exponent
)";
const string fragInput_u_matAmbi     = R"(
uniform vec4        u_matAmbi;                      // ambient color reflection coefficient (ka))";
const string fragInput_u_matDiff     = R"(
uniform vec4        u_matDiff;                      // diffuse color reflection coefficient (kd))";
const string fragInput_u_matEmis     = R"(
uniform vec4        u_matEmis;                      // emissive color (ke))";
const string fragInput_u_matRough    = R"(
uniform float       u_matRough;                     // roughness factor (0-1))";
const string fragInput_u_matMetal    = R"(
uniform float       u_matMetal;                     // metalness factor (0-1)";
//-----------------------------------------------------------------------------
const string fragInput_u_matTexDm       = R"(
uniform sampler2D   u_matTextureDiffuse0;           // Diffuse color map)";
const string fragInput_u_matTexNm       = R"(
uniform sampler2D   u_matTextureNormal0;            // Normal bump map)";
const string fragInput_u_matTexEm       = R"(
uniform sampler2D   u_matTextureEmissive0;          // PBR material emissive texture)";
const string fragInput_u_matTexOm       = R"(
uniform sampler2D   u_matTextureOcclusion0;         // Ambient occlusion map)";
const string fragInput_u_matTexRm       = R"(
uniform sampler2D   u_matTextureRoughness0;         // PBR material roughness texture)";
const string fragInput_u_matTexMm       = R"(
uniform sampler2D   u_matTextureMetallic0;          // PBR material metallic texture)";
const string fragInput_u_matTexRmMm     = R"(
uniform sampler2D   u_matTextureRoughMetal0;        // PBR material roughness-metallic texture)";
const string fragInput_u_matTexOmRmMm   = R"(
uniform sampler2D   u_matTextureOccluRoughMetal0;   // PBR material occlusion-roughness-metalic texture)";
const string fragInput_u_matGetsSm      = R"(

uniform bool        u_matGetsShadows;               // flag if material receives shadows)";
const string fragInput_u_skyCookEnvMaps = R"(
uniform samplerCube u_skyIrradianceCubemap; // PBR skybox irradiance light
uniform samplerCube u_skyRoughnessCubemap;  // PBR skybox cubemap for rough reflections
uniform sampler2D   u_skyBrdfLutTexture;    // PBR lighting lookup table for BRDF
uniform float       u_skyExposure;          // PBR skybox exposure)";
//-----------------------------------------------------------------------------
const string fragInput_u_cam = R"(

uniform int         u_camProjType;    // type of stereo
uniform int         u_camStereoEye;     // -1=left, 0=center, 1=right
uniform mat3        u_camStereoColors;  // color filter matrix
uniform bool        u_camFogIsOn;       // flag if fog is on
uniform int         u_camFogMode;       // 0=LINEAR, 1=EXP, 2=EXP2
uniform float       u_camFogDensity;    // fog density value
uniform float       u_camFogStart;      // fog start distance
uniform float       u_camFogEnd;        // fog end distance
uniform vec4        u_camFogColor;      // fog color (usually the background)
uniform float       u_camClipNear;      // camera near plane
uniform float       u_camClipFar;       // camera far plane
uniform float       u_camBkgdWidth;     // camera background width
uniform float       u_camBkgdHeight;    // camera background height
uniform float       u_camBkgdLeft;      // camera background left
uniform float       u_camBkgdBottom;    // camera background bottom)";
//-----------------------------------------------------------------------------
const string fragOutputs_o_fragColor = R"(

out     vec4        o_fragColor;        // output fragment color)";
//-----------------------------------------------------------------------------
const string fragFunctionsLightingBlinnPhong = R"(
//-----------------------------------------------------------------------------
void directLightBlinnPhong(in    int  i,         // Light number between 0 and NUM_LIGHTS
                           in    vec3 N,         // Normalized normal at v_P
                           in    vec3 E,         // Normalized direction at v_P to the eye
                           in    vec3 S,         // Normalized light spot direction
                           in    float shadow,   // shadow factor
                           inout vec4 Ia,        // Ambient light intensity
                           inout vec4 Id,        // Diffuse light intensity
                           inout vec4 Is)        // Specular light intensity
{
    // Calculate diffuse & specular factors
    float diffFactor = max(dot(N, S), 0.0);
    float specFactor = 0.0;

    if (diffFactor!=0.0)
    {
        vec3 H = normalize(S + E);// Half vector H between S and E
        specFactor = pow(max(dot(N, H), 0.0), u_matShin);
    }

    // accumulate directional light intensities w/o attenuation
    Ia += u_lightAmbi[i];
    Id += u_lightDiff[i] * diffFactor * (1.0 - shadow);
    Is += u_lightSpec[i] * specFactor * (1.0 - shadow);
}
//-----------------------------------------------------------------------------
void pointLightBlinnPhong( in    int   i,
                           in    vec3  N,
                           in    vec3  E,
                           in    vec3  S,
                           in    vec3  L,
                           in    float shadow,
                           inout vec4  Ia,
                           inout vec4  Id,
                           inout vec4  Is)
{
    // Calculate attenuation over distance & normalize L
    float att = 1.0;
    if (u_lightDoAtt[i])
    {
        vec3 att_dist;
        att_dist.x = 1.0;
        att_dist.z = dot(L, L);// = distance * distance
        att_dist.y = sqrt(att_dist.z);// = distance
        att = min(1.0 / dot(att_dist, u_lightAtt[i]), 1.0);
        L /= att_dist.y;// = normalize(L)
    }
    else
        L = normalize(L);

    // Calculate diffuse & specular factors
    vec3 H = normalize(E + L);              // Blinn's half vector is faster than Phongs reflected vector
    float diffFactor = max(dot(N, L), 0.0); // Lambertian downscale factor for diffuse reflection
    float specFactor = 0.0;
    if (diffFactor!=0.0)    // specular reflection is only possible if surface is lit from front
        specFactor = pow(max(dot(N, H), 0.0), u_matShin); // specular shininess

    // Calculate spot attenuation
    float spotAtt = 1.0;// Spot attenuation
    if (u_lightSpotDeg[i] < 180.0)
    {
        float spotDot;// Cosine of angle between L and spotdir
        spotDot = dot(-L, S);
        if (spotDot < u_lightSpotCos[i])  // if outside spot cone
            spotAtt = 0.0;
        else
            spotAtt = max(pow(spotDot, u_lightSpotExp[i]), 0.0);
    }

    // Accumulate light intensities
    Ia += att * u_lightAmbi[i];
    Id += att * spotAtt * u_lightDiff[i] * diffFactor * (1.0 - shadow);
    Is += att * spotAtt * u_lightSpec[i] * specFactor * (1.0 - shadow);
})";
//-----------------------------------------------------------------------------
const string fragFunctionsLightingCookTorrance = R"(
//-----------------------------------------------------------------------------
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
// ----------------------------------------------------------------------------
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}
//-----------------------------------------------------------------------------
float distributionGGX(vec3 N, vec3 H, float roughness)
{
    float PI     = 3.14159265;
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}
//-----------------------------------------------------------------------------
float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
//-----------------------------------------------------------------------------
float geometrySmith(vec3 N, vec3 E, vec3 L, float roughness)
{
    float NdotV = max(dot(N, E), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = geometrySchlickGGX(NdotV, roughness);
    float ggx1  = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}
//-----------------------------------------------------------------------------
void directLightCookTorrance(in    int   i,        // Light index
                             in    vec3  N,        // Normalized normal at v_P_VS
                             in    vec3  E,        // Normalized vector from v_P to the eye
                             in    vec3  S,        // Normalized light spot direction
                             in    vec3  F0,       // Fresnel reflection at 90 deg. (0 to N)
                             in    vec3  matDiff,  // diffuse material reflection
                             in    float matMetal, // diffuse material reflection
                             in    float matRough, // diffuse material reflection
                             in    float shadow,   // shadow factor (0.0 - 1.0)
                             inout vec3  Lo)       // reflected intensity
{
    float PI = 3.14159265;
    vec3 H = normalize(E + S);  // Normalized halfvector between eye and light vector

    vec3 radiance = u_lightDiff[i].rgb;  // Per light radiance without attenuation

    // cook-torrance brdf
    float NDF = distributionGGX(N, H, matRough);
    float G   = geometrySmith(N, E, S, matRough);
    vec3  F   = fresnelSchlick(max(dot(H, E), 0.0), F0);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - matMetal;

    vec3  nominator   = NDF * G * F;
    float denominator = 4.0 * max(dot(N, E), 0.0) * max(dot(N, S), 0.0) + 0.001;
    vec3  specular    = nominator / denominator;

    // add to outgoing radiance Lo
    float NdotL = max(dot(N, S), 0.0);

    Lo += (kD*matDiff.rgb/PI + specular) * radiance * NdotL * (1.0 - shadow);
}
//-----------------------------------------------------------------------------
void pointLightCookTorrance(in    int   i,        // Light index
                            in    vec3  N,        // Normalized normal at v_P_VS
                            in    vec3  E,        // Normalized vector from v_P to the eye
                            in    vec3  L,        // Vector from v_P to the light
                            in    vec3  S,        // Normalized light spot direction
                            in    vec3  F0,       // Fresnel reflection at 90 deg. (0 to N)
                            in    vec3  matDiff,  // diffuse material reflection
                            in    float matMetal, // diffuse material reflection
                            in    float matRough, // diffuse material reflection
                            in    float shadow,   // shadow factor (0.0 - 1.0)
                            inout vec3  Lo)       // reflected intensity
{
    float PI = 3.14159265;
    float distance = length(L); // distance to light
    L /= distance;              // normalize light vector
    float att = 1.0 / (distance*distance);  // quadratic light attenuation

    // Calculate spot attenuation
    if (u_lightSpotDeg[i] < 180.0)
    {
        float spotAtt; // Spot attenuation
        float spotDot; // Cosine of angle between L and spotdir
        spotDot = dot(-L, S);
        if (spotDot < u_lightSpotCos[i]) spotAtt = 0.0;
        else spotAtt = max(pow(spotDot, u_lightSpotExp[i]), 0.0);
        att *= spotAtt;
    }

    vec3 radiance = u_lightDiff[i].rgb * att;  // per light radiance

    // cook-torrance brdf
    vec3  H   = normalize(E + L);  // Normalized halfvector between eye and light vector
    float NDF = distributionGGX(N, H, matRough);
    float G   = geometrySmith(N, E, L, matRough);
    vec3  F   = fresnelSchlick(max(dot(H, E), 0.0), F0);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - matMetal;

    vec3  nominator   = NDF * G * F;
    float denominator = 4.0 * max(dot(N, E), 0.0) * max(dot(N, L), 0.0) + 0.001;
    vec3  specular    = nominator / denominator;

    // add to outgoing radiance Lo
    float NdotL = max(dot(N, L), 0.0);

    Lo += (kD*matDiff.rgb/PI + specular) * radiance * NdotL * (1.0 - shadow);
})";
//-----------------------------------------------------------------------------
const string fragFunctionDoStereoSeparation = R"(
//-----------------------------------------------------------------------------
void doStereoSeparation()
{
    // See SLProjType in SLEnum.h
    if (u_camProjType > 8) // stereoColors
    {
        // Apply color filter but keep alpha
        o_fragColor.rgb = u_camStereoColors * o_fragColor.rgb;
    }
    else if (u_camProjType == 6) // stereoLineByLine
    {
        if (mod(floor(gl_FragCoord.y), 2.0) < 0.5)// even
        {
            if (u_camStereoEye ==-1)
                discard;
        } else // odd
        {
            if (u_camStereoEye == 1)
                discard;
        }
    }
    else if (u_camProjType == 7) // stereoColByCol
    {
        if (mod(floor(gl_FragCoord.x), 2.0) < 0.5)// even
        {
            if (u_camStereoEye ==-1)
                discard;
        } else // odd
        {
            if (u_camStereoEye == 1)
                discard;
        }
    }
    else if (u_camProjType == 8) // stereoCheckerBoard
    {
        bool h = (mod(floor(gl_FragCoord.x), 2.0) < 0.5);
        bool v = (mod(floor(gl_FragCoord.y), 2.0) < 0.5);
        if (h==v)// both even or odd
        {
            if (u_camStereoEye ==-1)
                discard;
        } else // odd
        {
            if (u_camStereoEye == 1)
                discard;
        }
    }
})";
//-----------------------------------------------------------------------------
const string fragFunctionFogBlend = R"(
//-----------------------------------------------------------------------------
vec4 fogBlend(vec3 P_VS, vec4 inColor)
{
    float factor = 0.0f;
    float distance = length(P_VS);

    switch (u_camFogMode)
    {
        case 0:
            factor = (u_camFogEnd - distance) / (u_camFogEnd - u_camFogStart);
            break;
        case 1:
            factor = exp(-u_camFogDensity * distance);
            break;
        default:
            factor = exp(-u_camFogDensity * distance * u_camFogDensity * distance);
            break;
    }

    vec4 outColor = factor * inColor + (1.0 - factor) * u_camFogColor;
    outColor = clamp(outColor, 0.0, 1.0);
    return outColor;
})";
//-----------------------------------------------------------------------------
const string fragFunctionDoColoredShadows = R"(
//-----------------------------------------------------------------------------
void doColoredShadows(in vec3 N)
{
    const vec3 SHADOW_COLOR[6] = vec3[6](vec3(1.0, 0.0, 0.0),
                                         vec3(0.0, 1.0, 0.0),
                                         vec3(0.0, 0.0, 1.0),
                                         vec3(1.0, 1.0, 0.0),
                                         vec3(0.0, 1.0, 1.0),
                                         vec3(1.0, 0.0, 1.0));

    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            if (u_lightPosVS[i].w == 0.0)
            {
                // We use the spot light direction as the light direction vector
                vec3 S = normalize(-u_lightSpotDir[i].xyz);

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, S) : 0.0;
                if (u_lightNumCascades[i] > 0)
                {
                    int casIndex = getCascadesDepthIndex(i, u_lightNumCascades[i]);
                    o_fragColor.rgb += shadow * SHADOW_COLOR[casIndex];
                } else
                    o_fragColor.rgb += shadow * SHADOW_COLOR[0];
            }
            else
            {
                vec3 L = u_lightPosVS[i].xyz - v_P_VS; // Vector from v_P to light in VS

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, L) : 0.0;
                o_fragColor.rgb += shadow * SHADOW_COLOR[0];
            }
        }
    }
})";
//-----------------------------------------------------------------------------
const string fragMain_Begin         = R"(
//-----------------------------------------------------------------------------
void main()
{)";
const string fragMain_0_Intensities = R"(
    vec4 Ia = vec4(0.0); // Accumulated ambient light intensity at v_P_VS
    vec4 Id = vec4(0.0); // Accumulated diffuse light intensity at v_P_VS
    vec4 Is = vec4(0.0); // Accumulated specular light intensity at v_P_VS
)";
//-----------------------------------------------------------------------------
const string fragMain_1_EN_in_VS    = R"(
    vec3 E = normalize(-v_P_VS); // Interpolated vector from p to the eye
    vec3 N = normalize(v_N_VS);  // A input normal has not anymore unit length
)";
const string fragMain_1_EN_in_TS    = R"(
    vec3 E = normalize(v_eyeDirTS);   // normalized interpolated eye direction
    // Get normal from normal map, move from [0,1] to [-1, 1] range & normalize
    vec3 N = normalize(texture(u_matTextureNormal0, v_uv0).rgb * 2.0 - 1.0);
)";
const string fragMain_1_matEmis     = R"(
    vec4  matEmis  = u_matEmis;)";
const string fragMain_1_matEmis_Em  = R"(
    vec4  matEmis  = texture(u_matTextureEmissive0, v_uv0);)";
const string fragMain_1_matOccl     = R"(
    float matOccl  = 1.0;)";
const string fragMain_1_matOccl_Om0 = R"(
    float matOccl  = texture(u_matTextureOcclusion0, v_uv0).r;)";
const string fragMain_1_matOccl_Om1 = R"(
    float matOccl  = texture(u_matTextureOcclusion0, v_uv1).r;)";
//-----------------------------------------------------------------------------
const string fragMainBlinn_2_LightLoop     = R"(

    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            if (u_lightPosVS[i].w == 0.0)
            {
                // We use the spot light direction as the light direction vector
                vec3 S = normalize(-u_lightSpotDir[i].xyz);
                directLightBlinnPhong(i, N, E, S, 0.0, Ia, Id, Is);
            }
            else
            {
                vec3 S = u_lightSpotDir[i]; // normalized spot direction in VS
                vec3 L = u_lightPosVS[i].xyz - v_P_VS; // Vector from v_P to light in VS
                pointLightBlinnPhong(i, N, E, S, L, 0.0, Ia, Id, Is);
            }
        }
    }
)";
const string fragMainBlinn_2_LightLoopNm   = R"(

    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            if (u_lightPosVS[i].w == 0.0)
            {
                // We use the spot light direction as the light direction vector
                vec3 S = normalize(-v_spotDirTS[i]);
                directLightBlinnPhong(i, N, E, S, 0.0, Ia, Id, Is);
            }
            else
            {
                vec3 S = normalize(v_spotDirTS[i]); // normalized spot direction in TS
                vec3 L = v_lightDirTS[i]; // Vector from v_P to light in TS
                pointLightBlinnPhong(i, N, E, S, L, 0.0, Ia, Id, Is);
            }
        }
    }
)";
const string fragMainBlinn_2_LightLoopSm   = R"(

    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            if (u_lightPosVS[i].w == 0.0)
            {
                // We use the spot light direction as the light direction vector
                vec3 S = normalize(-u_lightSpotDir[i].xyz);

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, S) : 0.0;
                directLightBlinnPhong(i, N, E, S, shadow, Ia, Id, Is);
            }
            else
            {
                vec3 S = u_lightSpotDir[i]; // normalized spot direction in VS
                vec3 L = u_lightPosVS[i].xyz - v_P_VS; // Vector from v_P to light in VS

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, L) : 0.0;
                pointLightBlinnPhong(i, N, E, S, L, shadow, Ia, Id, Is);
            }
        }
    }
)";
const string fragMainBlinn_2_LightLoopNmSm = R"(

    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            if (u_lightPosVS[i].w == 0.0)
            {
                // We use the spot light direction as the light direction vector
                vec3 S = normalize(-v_spotDirTS[i]);

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, S) : 0.0;
                directLightBlinnPhong(i, N, E, S, shadow, Ia, Id, Is);
            }
            else
            {
                vec3 S = normalize(v_spotDirTS[i]); // normalized spot direction in TS
                vec3 L = v_lightDirTS[i]; // Vector from v_P to light in TS

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, L) : 0.0;
                pointLightBlinnPhong(i, N, E, S, L, shadow, Ia, Id, Is);
            }
        }
    }
)";
const string fragMainBlinn_3_FragColor     = R"(
    // Sum up all the reflected color components
    o_fragColor =  u_matEmis +
                   u_globalAmbi +
                   Ia * u_matAmbi * matOccl +
                   Id * u_matDiff +
                   Is * u_matSpec;

    // For correct alpha blending overwrite alpha component
    o_fragColor.a = u_matDiff.a;
)";
const string fragMainBlinn_3_FragColorDm   = R"(
    // Sum up all the reflected color components
    o_fragColor =  u_matEmis +
                   u_globalAmbi +
                   Ia * u_matAmbi * matOccl +
                   Id * u_matDiff;

    // Componentwise multiply w. texture color
    o_fragColor *= texture(u_matTextureDiffuse0, v_uv0);

    // add finally the specular RGB-part
    vec4 specColor = Is * u_matSpec;
    o_fragColor.rgb += specColor.rgb;
)";
//-----------------------------------------------------------------------------
const string fragMain_4_ColoredShadows = R"(
    // Colorize cascaded shadows for debugging purpose
    if (u_lightsDoColoredShadows)
        doColoredShadows(N);
)";
//-----------------------------------------------------------------------------
const string fragMain_5_FogGammaStereo = R"(
    // Apply fog by blending over distance
    if (u_camFogIsOn)
        o_fragColor = fogBlend(v_P_VS, o_fragColor);

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));

    // Apply stereo eye separation
    if (u_camProjType > 1)
        doStereoSeparation();
}
)";
//-----------------------------------------------------------------------------
const string fragMainCook_1_matDiff       = R"(
    vec4  matDiff  = u_matDiff;)";
const string fragMainCook_1_matDiff_Dm    = R"(
    vec4  matDiff  = pow(texture(u_matTextureDiffuse0, v_uv0), vec4(2.2));)";
const string fragMainCook_1_matEmis       = R"(
    vec4  matEmis  = u_matEmis;)";
const string fragMainCook_1_matEmis_Em    = R"(
    vec4  matEmis  = pow(texture(u_matTextureEmissive0, v_uv0), vec4(2.2));)";
const string fragMainCook_1_matRough      = R"(
    float matRough = u_matRough;)";
const string fragMainCook_1_matRough_Rm   = R"(
    float matRough = texture(u_matTextureRoughness0, v_uv0).r;)";
const string fragMainCook_1_matRough_RMm  = R"(
     float matRough = texture(u_matTextureRoughMetal0, v_uv0).g;)";
const string fragMainCook_1_matRough_ORMm = R"(
    float matRough = texture(u_matTextureOccluRoughMetal0, v_uv0).g;)";
const string fragMainCook_1_matMetal      = R"(
     float matMetal = u_matMetal;)";
const string fragMainCook_1_matMetal_Mm   = R"(
     float matMetal = texture(u_matTextureMetallic0, v_uv0).r;)";
const string fragMainCook_1_matMetal_RMm  = R"(
     float matMetal = texture(u_matTextureRoughMetal0, v_uv0).b;)";
const string fragMainCook_1_matMetal_ORMm = R"(
    float matMetal = texture(u_matTextureOccluRoughMetal0, v_uv0).b;)";
const string fragMainCook_1_matOcclu_1    = R"(
     float matOccl  = 1.0;)";
const string fragMainCook_1_matOcclu_Om   = R"(
     float matOccl  = texture(u_matTextureOcclusion0, v_uv0).r;)";
const string fragMainCook_1_matOcclu_ORMm = R"(
    float matOccl  = texture(u_matTextureOccluRoughMetal0, v_uv0).r;)";
const string fragMainCook_2_LightLoop     = R"(
    // Init Fresnel reflection at 90 deg. (0 to N)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, matDiff.rgb, matMetal);

    // Get the reflection from all lights into Lo
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            if (u_lightPosVS[i].w == 0.0)
            {
                // We use the spot light direction as the light direction vector
                vec3 S = normalize(-u_lightSpotDir[i].xyz);
                directLightCookTorrance(i, N, E, S, F0,
                                        matDiff.rgb,
                                        matMetal,
                                        matRough,
                                        0.0,
                                        Lo);
            }
            else
            {
                vec3 L = u_lightPosVS[i].xyz - v_P_VS;
                vec3 S = u_lightSpotDir[i]; // normalized spot direction in VS
                pointLightCookTorrance( i, N, E, L, S, F0,
                                        matDiff.rgb,
                                        matMetal,
                                        matRough,
                                        0.0,
                                        Lo);
            }
        }
    }
)";
const string fragMainCook_2_LightLoopNm   = R"(
    // Init Fresnel reflection at 90 deg. (0 to N)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, matDiff.rgb, matMetal);

    // Get the reflection from all lights into Lo
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            if (u_lightPosVS[i].w == 0.0)
            {
                // We use the spot light direction as the light direction vector
                vec3 S = normalize(-v_spotDirTS[i]);
                directLightCookTorrance(i, N, E, S, F0,
                                        matDiff.rgb,
                                        matMetal,
                                        matRough,
                                        0.0,
                                        Lo);
            }
            else
            {
                vec3 L = v_lightDirTS[i]; // Vector from v_P to light in TS
                vec3 S = normalize(-v_spotDirTS[i]);
                pointLightCookTorrance( i, N, E, L, S, F0,
                                        matDiff.rgb,
                                        matMetal,
                                        matRough,
                                        0.0,
                                        Lo);
            }
        }
    }
)";
const string fragMainCook_2_LightLoopSm   = R"(
    // Init Fresnel reflection at 90 deg. (0 to N)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, matDiff.rgb, matMetal);

    // Get the reflection from all lights into Lo
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            if (u_lightPosVS[i].w == 0.0)
            {
                // We use the spot light direction as the light direction vector
                vec3 S = normalize(-u_lightSpotDir[i].xyz);

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, S) : 0.0;
                directLightCookTorrance(i, N, E, S, F0,
                                        matDiff.rgb,
                                        matMetal,
                                        matRough,
                                        shadow,
                                        Lo);
            }
            else
            {
                vec3 L = u_lightPosVS[i].xyz - v_P_VS;
                vec3 S = u_lightSpotDir[i]; // normalized spot direction in VS

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, L) : 0.0;
                pointLightCookTorrance( i, N, E, L, S, F0,
                                        matDiff.rgb,
                                        matMetal,
                                        matRough,
                                        shadow,
                                        Lo);
            }
        }
    }
)";
const string fragMainCook_2_LightLoopNmSm = R"(
    // Init Fresnel reflection at 90 deg. (0 to N)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, matDiff.rgb, matMetal);

    // Get the reflection from all lights into Lo
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            if (u_lightPosVS[i].w == 0.0)
            {
                // We use the spot light direction as the light direction vector
                vec3 S = normalize(-v_spotDirTS[i]);

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, S) : 0.0;
                directLightCookTorrance(i, N, E, S, F0,
                                        matDiff.rgb,
                                        matMetal,
                                        matRough,
                                        shadow,
                                        Lo);
            }
            else
            {
                vec3 L = v_lightDirTS[i]; // Vector from v_P to light in TS
                vec3 S = normalize(-v_spotDirTS[i]);

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, L) : 0.0;
                pointLightCookTorrance( i, N, E, L, S, F0,
                                        matDiff.rgb,
                                        matMetal,
                                        matRough,
                                        shadow,
                                        Lo);
            }
        }
    }
)";
const string fragMainCook_3_FragColor     = R"(

    // ambient lighting (note that the next IBL tutorial will replace
    // this ambient lighting with environment lighting).
    vec3 ambient = vec3(0.03) * matDiff.rgb * matOccl;

    vec3 color   = ambient + matEmis.rgb + Lo;

    // HDR tone-mapping
    color = color / (color + vec3(1.0));
    o_fragColor = vec4(color, 1.0);

    // For correct alpha blending overwrite alpha component
    o_fragColor.a = matDiff.a;
)";
const string fragMainCook_3_FragColorSky  = R"(

    // Build diffuse reflection from environment light map
    vec3 F = fresnelSchlickRoughness(max(dot(N, E), 0.0), F0, matRough);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - matMetal;
    vec3 irradiance = texture(u_skyIrradianceCubemap, N).rgb;
    vec3 diffuse    = kD * irradiance * matDiff.rgb;

    // sample both the pre-filter map and the BRDF lut and combine them together as per the split-sum approximation to get the IBL specular part.
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(u_skyRoughnessCubemap, v_R_OS, matRough * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(u_skyBrdfLutTexture, vec2(max(dot(N, E), 0.0), matRough)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);
    vec3 ambient = (diffuse + specular) * matOccl;

    vec3 color = ambient + matEmis.rgb + Lo;

    // Exposure tone mapping
    vec3 mapped = vec3(1.0) - exp(-color * u_skyExposure);
    o_fragColor = vec4(mapped, 1.0);

    // For correct alpha blending overwrite alpha component
    o_fragColor.a = matDiff.a;
)";
//-----------------------------------------------------------------------------
const string fragMainVideoBkgd = R"(
void main()
{
    float x = (gl_FragCoord.x - u_camBkgdLeft) / u_camBkgdWidth;
    float y = (gl_FragCoord.y - u_camBkgdBottom) / u_camBkgdHeight;

    if(x < 0.0f || y < 0.0f || x > 1.0f || y > 1.0f)
        o_fragColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    else
        o_fragColor = texture(u_matTextureDiffuse0, vec2(x, y));

    vec3 N = normalize(v_N_VS);  // A input normal has not anymore unit length
    float shadow = 0.0;

    // Colorize cascaded shadows for debugging purpose
    if (u_lightsDoColoredShadows)
        doColoredShadows(N);
    else
    {
        for (int i = 0; i < NUM_LIGHTS; ++i)
        {
            if (u_lightIsOn[i])
            {
                if (u_lightPosVS[i].w == 0.0)
                {
                    // We use the spot light direction as the light direction vector
                    vec3 S = normalize(-u_lightSpotDir[i].xyz);

                    // Test if the current fragment is in shadow
                    shadow = u_matGetsShadows ? shadowTest(i, N, S) : 0.0;
                }
                else
                {
                    vec3 L = u_lightPosVS[i].xyz - v_P_VS; // Vector from v_P to light in VS

                    // Test if the current fragment is in shadow
                    shadow = u_matGetsShadows ? shadowTest(i, N, L) : 0.0;
                }
                o_fragColor = o_fragColor * min(1.0 - shadow + u_matAmbi.r, 1.0);
            }
        }
    }

)";
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//! Builds unique program name that identifies shader program
/*! See the class information for more insights of the generated name. This
 function is used in advance of the code generation to check if the program
 already exists in the asset manager. See SLMaterial::activate.
 @param mat Parent material pointer
 @param lights Pointer of vector of lights
 @param programName Reference to program name string that gets built

 The shader program gets a unique name with the following pattern:
 <pre>
 genCook-D00-N00-E00-O01-RM00-Sky-C4s
    |    |   |   |   |   |    |   |
    |    |   |   |   |   |    |   + Directional light w. 4 shadow cascades
    |    |   |   |   |   |    + Ambient light from skybox
    |    |   |   |   |   + Roughness-metallic map with index 0 and uv 0
    |    |   |   |   + Ambient Occlusion map with index 0 and uv 1
    |    |   |   + Emissive Map with index 0 and uv 0
    |    |   + Normal Map with index 0 and uv 0
    |    + Diffuse Texture Mapping with index 0 and uv 0
    + Cook-Torrance or Blinn-Phong reflection model
 </pre>
 */
void SLGLProgramGenerated::buildProgramName(SLMaterial* mat,
                                            SLVLight*   lights,
                                            string&     programName)
{
    assert(mat && "No material pointer passed!");
    assert(lights && !lights->empty() && "No lights passed!");
    programName = "gen";

    if (mat->hasTextureType(TT_videoBkgd))
        programName += "VideoBkgdDm";
    else if (mat->reflectionModel() == RM_BlinnPhong)
        programName += "Blinn";
    else if (mat->reflectionModel() == RM_CookTorrance)
        programName += "Cook";
    else
        programName += "Custom";

    programName += mat->texturesString();
    programName += "-";

    // Add letter per light type
    for (auto light : *lights)
    {
        if (light->positionWS().w == 0.0f)
        {
            if (light->doCascadedShadows())
                programName += "C" + std::to_string(light->shadowMap()->numCascades()); // Directional light with cascaded shadowmap
            else
                programName += "D";                                                     // Directional light
        }
        else if (light->spotCutOffDEG() < 180.0f)
            programName += "S"; // Spot light
        else
            programName += "P"; // Point light
        if (light->createsShadows())
            programName += "s"; // Creates shadows
    }
}
//-----------------------------------------------------------------------------
/*! See the class information for more insights of the generated name. This
 function is used in advance of the code generation to check if the program
 already exists in the asset manager. See SLMaterial::activate.
 @param mat Parent material pointer
 @param programName Reference to program name string that gets built

 The shader program gets a unique name with the following pattern:
 <pre>
 genCook-D00-N00-E00-O01-RM00-Sky-C4s
    |    |   |   |   |   |    |   |
    |    |   |   |   |   |    |   + Directional light w. 4 shadow cascades
    |    |   |   |   |   |    + Ambient light from skybox
    |    |   |   |   |   + Roughness-metallic map with index 0 and uv 0
    |    |   |   |   + Ambient Occlusion map with index 0 and uv 1
    |    |   |   + Emissive Map with index 0 and uv 0
    |    |   + Normal Map with index 0 and uv 0
    |    + Diffuse Texture Mapping with index 0 and uv 0
    + Cook-Torrance or Blinn-Phong reflection model
 </pre>
 */
void SLGLProgramGenerated::buildProgramNamePS(SLMaterial* mat,
                                              string&     programName,
                                              bool        isDrawProg)
{
    assert(mat && "No material pointer passed!");
    programName = "gen";

    if (mat->reflectionModel() == RM_Particle)
        programName += "Particle";
    else
        programName += "Custom";

    // programName += "-";
    if (isDrawProg) // Drawing program
    {
        programName += "-Draw";
        programName += mat->texturesString();
        GLint billboardType = mat->ps()->billboardType();      // Billboard type (0 -> default; 1 -> vertical billboard, 2 -> horizontal billboard)
        bool  AlOvLi        = mat->ps()->doAlphaOverLT();      // Alpha over life
        bool  AlOvLiCu      = mat->ps()->doAlphaOverLTCurve(); // Alpha over life curve
        bool  SiOvLi        = mat->ps()->doSizeOverLT();       // Size over life
        bool  SiOvLiCu      = mat->ps()->doSizeOverLTCurve();  // Size over life curve
        bool  Co            = mat->ps()->doColor();            // Color over life
        bool  CoOvLi        = mat->ps()->doColorOverLT();      // Color over life
        bool  FlBoTex       = mat->ps()->doFlipBookTexture();  // Flipbook texture
        bool  WS            = mat->ps()->doWorldSpace();       // World space or local space
        bool  rot           = mat->ps()->doRotation();         // Rotation
        programName += "-B" + std::to_string(billboardType);
        if (rot) programName += "-RT";

        if (AlOvLi) programName += "-AL";
        if (AlOvLi && AlOvLiCu) programName += "cu";
        if (SiOvLi) programName += "-SL";
        if (SiOvLi && SiOvLiCu) programName += "cu";
        if (Co) programName += "-CO";
        if (Co && CoOvLi) programName += "cl";
        if (FlBoTex) programName += "-FB";
        if (WS) programName += "-WS";
    }
    else                                                  // Updating program
    {
        bool counterGap = mat->ps()->doCounterGap();      // Counter gap/lag
        bool acc        = mat->ps()->doAcc();             // Acceleration
        bool accDiffDir = mat->ps()->doAccDiffDir();      // Acceleration different direction
        bool gravity    = mat->ps()->doGravity();         // Gravity
        bool FlBoTex    = mat->ps()->doFlipBookTexture(); // Flipbook texture
        bool rot        = mat->ps()->doRotation();        // Rotation
        bool rotRange   = mat->ps()->doRotRange();        // Rotation range
        bool shape      = mat->ps()->doShape();           // Shape
        programName += "-Update";
        if (counterGap) programName += "-CG";
        if (rot) programName += "-RT";
        if (rot) programName += rotRange ? "ra" : "co";
        if (acc)
        {
            programName += "-AC";
            programName += accDiffDir ? "di" : "co";
        }
        if (gravity) programName += "-GR";
        if (FlBoTex) programName += "-FB";
        if (shape) programName += "-SH";
    }
}
//-----------------------------------------------------------------------------
/*! Builds the GLSL program code for the vertex and fragment shaders. The code
 * is only assembled but not compiled and linked. This happens within the
 * before the first draw call from within SLMesh::draw.
 * @param mat Parent material pointer
 * @param lights Pointer of vector of lights
 */
void SLGLProgramGenerated::buildProgramCode(SLMaterial* mat,
                                            SLVLight*   lights)
{
    if (mat->name() == "IBLMat")
    {
        std::cout << "build program code for IBLMat" << std::endl;
    }
    assert(mat && "No material pointer passed!");
    assert(!lights->empty() && "No lights passed!");
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Check what textures the material has
    bool Dm  = mat->hasTextureType(TT_diffuse);   // Texture Mapping
    bool Nm  = mat->hasTextureType(TT_normal);    // Normal Mapping
    bool Hm  = mat->hasTextureType(TT_height);    // Height Mapping
    bool Om  = mat->hasTextureType(TT_occlusion); // Ambient Occlusion Mapping
    bool Vm  = mat->hasTextureType(TT_videoBkgd); // Video Background Mapping
    bool env = mat->skybox() != nullptr;          // Environment Mapping from skybox

    // Check if any of the scene lights does shadow mapping
    bool Sm = lightsDoShadowMapping(lights);

    if (mat->reflectionModel() == RM_BlinnPhong)
    {
        buildPerPixBlinn(mat, lights);
    }
    else if (mat->reflectionModel() == RM_CookTorrance)
    {
        buildPerPixCook(mat, lights);
    }
    else if (mat->reflectionModel() == RM_Custom)
    {
        if (Vm && Sm)
            buildPerPixVideoBkgdSm(lights);
        else
            SL_EXIT_MSG("SLGLProgramGenerated::buildProgramCode: Unknown program for RM_Custom.");
    }
    else
        SL_EXIT_MSG("SLGLProgramGenerated::buildProgramCode: Unknown Lighting Model.");
}
//-----------------------------------------------------------------------------
/*! Builds the GLSL program code for the vertex, geometry and fragment shaders
 * (for particle system drawing). The code is only assembled but not compiled and linked.
 * This happens within the before the first draw call from within SLMesh::draw.
 * @param mat Parent material pointer
 */
void SLGLProgramGenerated::buildProgramCodePS(SLMaterial* mat, bool isDrawProg)
{
    if (mat->name() == "IBLMat")
    {
        std::cout << "build program code for IBLMat" << std::endl;
    }
    assert(mat && "No material pointer passed!");
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    if (isDrawProg)
        buildPerPixParticle(mat);
    else
        buildPerPixParticleUpdate(mat);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixCook(SLMaterial* mat, SLVLight* lights)
{
    assert(mat && lights);
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Check what textures the material has
    bool Dm   = mat->hasTextureType(TT_diffuse);
    bool Nm   = mat->hasTextureType(TT_normal);
    bool Hm   = mat->hasTextureType(TT_height);
    bool Rm   = mat->hasTextureType(TT_roughness);
    bool Mm   = mat->hasTextureType(TT_metallic);
    bool RMm  = mat->hasTextureType(TT_roughMetal);
    bool Em   = mat->hasTextureType(TT_emissive);
    bool Om   = mat->hasTextureType(TT_occlusion);
    bool ORMm = mat->hasTextureType(TT_occluRoughMetal);
    bool Vm   = mat->hasTextureType(TT_videoBkgd);
    bool Sm   = lightsDoShadowMapping(lights);
    bool uv0  = mat->usesUVIndex(0);
    bool uv1  = mat->usesUVIndex(1);
    bool sky  = mat->skybox() != nullptr;

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());

    // Vertex shader inputs
    vertCode += vertInput_a_pn;
    if (uv0) vertCode += vertInput_a_uv0;
    if (Nm) vertCode += vertInput_a_tangent;
    vertCode += vertInput_u_matrices_all;
    // if (sky) vertCode += vertInput_u_matrix_invMv;
    if (Nm) vertCode += vertInput_u_lightNm;

    // Vertex shader outputs
    vertCode += vertOutput_v_P_VS;
    if (Sm) vertCode += vertOutput_v_P_WS;
    vertCode += vertOutput_v_N_VS;
    if (sky) vertCode += vertOutput_v_R_OS;
    if (uv0) vertCode += vertOutput_v_uv0;
    if (Nm) vertCode += vertOutput_v_lightVecTS;

    // Vertex shader main loop
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    if (Sm) vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_N_VS;
    if (sky) vertCode += vertMain_v_R_OS;
    if (uv0) vertCode += vertMain_v_uv0;
    if (Nm) vertCode += vertMain_TBN_Nm;
    vertCode += vertMain_EndAll;

    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());

    // Fragment shader inputs
    fragCode += fragInput_v_P_VS;
    if (Sm) fragCode += fragInput_v_P_WS;
    fragCode += fragInput_v_N_VS;
    if (sky) fragCode += fragInput_v_R_OS;
    if (uv0) fragCode += fragInput_v_uv0;
    if (Nm) fragCode += fragInput_v_lightVecTS;

    // Fragment shader uniforms
    fragCode += fragInput_u_lightAll;
    if (Sm) fragCode += fragInput_u_lightSm(lights);
    fragCode += Dm ? fragInput_u_matTexDm : fragInput_u_matDiff;
    fragCode += Em ? fragInput_u_matTexEm : fragInput_u_matEmis;
    if (Rm) fragCode += fragInput_u_matTexRm;
    if (Mm) fragCode += fragInput_u_matTexMm;
    if (RMm) fragCode += fragInput_u_matTexRmMm;
    if (ORMm) fragCode += fragInput_u_matTexOmRmMm;
    if (!Rm && !RMm && !ORMm) fragCode += fragInput_u_matRough;
    if (!Mm && !RMm && !ORMm) fragCode += fragInput_u_matMetal;
    if (Nm) fragCode += fragInput_u_matTexNm;
    if (Om) fragCode += fragInput_u_matTexOm;
    if (Sm) fragCode += fragInput_u_matGetsSm;
    if (Sm) fragCode += fragInput_u_shadowMaps(lights);
    if (sky) fragCode += fragInput_u_skyCookEnvMaps;
    fragCode += fragInput_u_cam;

    // Fragment shader outputs
    fragCode += fragOutputs_o_fragColor;

    // Fragment shader functions
    fragCode += fragFunctionsLightingCookTorrance;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    if (Sm) fragCode += fragFunctionShadowTest(lights);
    if (Sm) fragCode += fragFunctionDoColoredShadows;

    // Fragment shader main loop
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += Nm ? fragMain_1_EN_in_TS : fragMain_1_EN_in_VS;
    fragCode += Dm ? fragMainCook_1_matDiff_Dm : fragMainCook_1_matDiff;
    fragCode += Em ? fragMainCook_1_matEmis_Em : fragMainCook_1_matEmis;
    fragCode += ORMm ? fragMainCook_1_matRough_ORMm : RMm ? fragMainCook_1_matRough_RMm
                                                    : Rm  ? fragMainCook_1_matRough_Rm
                                                          : fragMainCook_1_matRough;
    fragCode += ORMm ? fragMainCook_1_matMetal_ORMm : RMm ? fragMainCook_1_matMetal_RMm
                                                    : Rm  ? fragMainCook_1_matMetal_Mm
                                                          : fragMainCook_1_matMetal;
    fragCode += ORMm ? fragMainCook_1_matOcclu_ORMm : Om ? fragMainCook_1_matOcclu_Om
                                                         : fragMainCook_1_matOcclu_1;
    fragCode += Nm && Sm ? fragMainCook_2_LightLoopNmSm : Nm ? fragMainCook_2_LightLoopNm
                                                        : Sm ? fragMainCook_2_LightLoopSm
                                                             : fragMainCook_2_LightLoop;
    fragCode += sky ? fragMainCook_3_FragColorSky : fragMainCook_3_FragColor;
    if (Sm) fragCode += fragMain_4_ColoredShadows;
    fragCode += fragMain_5_FogGammaStereo;

    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinn(SLMaterial* mat, SLVLight* lights)
{
    assert(mat && lights);
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Check what textures the material has
    bool Dm  = mat->hasTextureType(TT_diffuse);
    bool Nm  = mat->hasTextureType(TT_normal);
    bool Hm  = mat->hasTextureType(TT_height);
    bool Em  = mat->hasTextureType(TT_emissive);
    bool Om0 = mat->hasTextureTypeWithUVIndex(TT_occlusion, 0, 0);
    bool Om1 = mat->hasTextureTypeWithUVIndex(TT_occlusion, 0, 1);
    bool Sm  = lightsDoShadowMapping(lights);
    bool uv0 = mat->usesUVIndex(0);
    bool uv1 = mat->usesUVIndex(1);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());

    // Vertex shader inputs
    vertCode += vertInput_a_pn;
    if (uv0) vertCode += vertInput_a_uv0;
    if (uv1) vertCode += vertInput_a_uv1;
    if (Nm) vertCode += vertInput_a_tangent;
    vertCode += vertInput_u_matrices_all;
    if (Nm) vertCode += vertInput_u_lightNm;

    // Vertex shader outputs
    vertCode += vertOutput_v_P_VS;
    if (Sm) vertCode += vertOutput_v_P_WS;
    vertCode += vertOutput_v_N_VS;
    if (uv0) vertCode += vertOutput_v_uv0;
    if (uv1) vertCode += vertOutput_v_uv1;
    if (Nm) vertCode += vertOutput_v_lightVecTS;

    // Vertex shader main loop
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    if (Sm) vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_N_VS;
    if (uv0) vertCode += vertMain_v_uv0;
    if (uv1) vertCode += vertMain_v_uv1;
    if (Nm) vertCode += vertMain_TBN_Nm;
    vertCode += vertMain_EndAll;

    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());

    // Fragment shader inputs
    fragCode += fragInput_v_P_VS;
    if (Sm) fragCode += fragInput_v_P_WS;
    fragCode += fragInput_v_N_VS;
    if (uv0) fragCode += fragInput_v_uv0;
    if (uv1) fragCode += fragInput_v_uv1;
    if (Nm) fragCode += fragInput_v_lightVecTS;

    // Fragment shader uniforms
    fragCode += fragInput_u_lightAll;
    fragCode += fragInput_u_matBlinnAll;
    if (Sm) fragCode += fragInput_u_lightSm(lights);
    if (Dm) fragCode += fragInput_u_matTexDm;
    if (Nm) fragCode += fragInput_u_matTexNm;
    if (Em) fragCode += fragInput_u_matTexEm;
    if (Om0 || Om1) fragCode += fragInput_u_matTexOm;
    if (Sm) fragCode += fragInput_u_matGetsSm;
    if (Sm) fragCode += fragInput_u_shadowMaps(lights);
    fragCode += fragInput_u_cam;

    // Fragment shader outputs
    fragCode += fragOutputs_o_fragColor;

    // Fragment shader functions
    fragCode += fragFunctionsLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    if (Sm) fragCode += fragFunctionShadowTest(lights);
    if (Sm) fragCode += fragFunctionDoColoredShadows;

    // Fragment shader main loop
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += Nm ? fragMain_1_EN_in_TS : fragMain_1_EN_in_VS;
    fragCode += Em ? fragMain_1_matEmis_Em : fragMain_1_matEmis;

    fragCode += Om0 ? fragMain_1_matOccl_Om0 : Om1 ? fragMain_1_matOccl_Om1
                                                   : fragMain_1_matOccl;

    fragCode += Nm && Sm ? fragMainBlinn_2_LightLoopNmSm : Nm ? fragMainBlinn_2_LightLoopNm
                                                         : Sm ? fragMainBlinn_2_LightLoopSm
                                                              : fragMainBlinn_2_LightLoop;
    fragCode += Dm ? fragMainBlinn_3_FragColorDm : fragMainBlinn_3_FragColor;
    if (Sm) fragCode += fragMain_4_ColoredShadows;
    fragCode += fragMain_5_FogGammaStereo;

    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixParticle(SLMaterial* mat)
{
    assert(_shaders.size() > 2 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment &&
           _shaders[2]->type() == ST_geometry);

    // Check what textures the material has
    bool  Dm            = mat->hasTextureType(TT_diffuse);
    GLint billboardType = mat->ps()->billboardType();      // Billboard type (0 -> default; 1 -> vertical billboard, 2 -> horizontal billboard)
    bool  rot           = mat->ps()->doRotation();         // Rotation
    bool  AlOvLi        = mat->ps()->doAlphaOverLT();      // Alpha over life
    bool  Co            = mat->ps()->doColor();            // Color over life
    bool  CoOvLi        = mat->ps()->doColorOverLT();      // Color over life
    bool  AlOvLiCu      = mat->ps()->doAlphaOverLTCurve(); // Alpha over life curve
    bool  SiOvLi        = mat->ps()->doSizeOverLT();       // Size over life
    bool  SiOvLiCu      = mat->ps()->doSizeOverLTCurve();  // Size over life curve
    bool  FlBoTex       = mat->ps()->doFlipBookTexture();  // Flipbook texture

    //////////////////////////////
    // Assemble vertex shader code
    //////////////////////////////

    string vertCode;
    vertCode += shaderHeader();

    // Vertex shader inputs
    vertCode += vertInput_PS_a_p;
    vertCode += vertInput_PS_a_st;
    if (rot) vertCode += vertInput_PS_a_r;
    if (FlBoTex) vertCode += vertInput_PS_a_texNum;

    // Vertex shader uniforms
    vertCode += vertInput_PS_u_time;
    vertCode += vertInput_u_matrix_vOmv;
    if (AlOvLi && AlOvLiCu) vertCode += vertInput_PS_u_al_bernstein_alpha;
    if (SiOvLi && SiOvLiCu) vertCode += vertInput_PS_u_al_bernstein_size;
    if (Co && CoOvLi) vertCode += vertInput_PS_u_colorOvLF;

    // Vertex shader outputs
    vertCode += vertOutput_PS_struct_Begin;
    vertCode += vertOutput_PS_struct_t;
    if (rot) vertCode += vertOutput_PS_struct_r;
    if (SiOvLi) vertCode += vertOutput_PS_struct_s;
    if (Co && CoOvLi) vertCode += vertOutput_PS_struct_c;
    if (FlBoTex) vertCode += vertOutput_PS_struct_texNum;
    vertCode += vertOutput_PS_struct_End;

    // Vertex shader functions
    if (Co && CoOvLi) vertCode += vertFunction_PS_ColorOverLT;

    // Vertex shader main loop
    vertCode += vertMain_Begin;
    vertCode += vertMain_PS_v_a;
    if (AlOvLi)
        vertCode += vertMain_PS_v_t_begin;
    else
        vertCode += vertMain_PS_v_t_default;
    if (AlOvLi) vertCode += AlOvLiCu ? vertMain_PS_v_t_curve : vertMain_PS_v_t_linear;
    if (AlOvLi) vertCode += vertMain_PS_v_t_end;
    if (rot) vertCode += vertMain_PS_v_r;
    if (SiOvLi) vertCode += vertMain_PS_v_s;
    if (SiOvLi && SiOvLiCu) vertCode += vertMain_PS_v_s_curve;
    if (Co && CoOvLi) vertCode += vertMain_PS_v_doColorOverLT;
    if (FlBoTex) vertCode += vertMain_PS_v_texNum;
    if (billboardType == BT_Vertical || billboardType == BT_Horizontal)
        vertCode += vertMain_PS_EndAll_VertBillboard;
    else
        vertCode += vertMain_PS_EndAll;

    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    ////////////////////////////////
    // Assemble geometry shader code
    ////////////////////////////////

    string geomCode;
    geomCode += shaderHeader();

    // geometry shader inputs
    geomCode += geomConfig_PS;

    geomCode += geomInput_PS_struct_Begin;
    geomCode += geomInput_PS_struct_t;
    if (rot) geomCode += geomInput_PS_struct_r;
    if (SiOvLi) geomCode += geomInput_PS_struct_s;
    if (Co && CoOvLi) geomCode += geomInput_PS_struct_c;
    if (FlBoTex) geomCode += geomInput_PS_struct_texNum;
    geomCode += geomInput_PS_struct_End;

    // geometry shader uniforms
    geomCode += geomInput_PS_u_ScaRa;
    if (Co && !CoOvLi) geomCode += geomInput_PS_u_c;
    if (FlBoTex) geomCode += geomInput_PS_u_col;
    if (FlBoTex) geomCode += geomInput_PS_u_row;
    geomCode += geomInput_u_matrix_p;
    if (billboardType == BT_Vertical)
        geomCode += geomInput_u_matrix_vertBillboard;
    else if (billboardType == BT_Horizontal)
        geomCode += vertInput_u_matrix_vOmv;

    // geometry shader outputs
    geomCode += geomOutput_PS_v_pC;
    geomCode += geomOutput_PS_v_tC;

    // geometry shader main loop
    geomCode += geomMain_PS_Begin;
    geomCode += geomMain_PS_v_s;
    if (SiOvLi) geomCode += geomMain_PS_v_sS;
    geomCode += geomMain_PS_v_rad;
    geomCode += geomMain_PS_v_p;
    geomCode += rot ? geomMain_PS_v_rot : geomMain_PS_v_rotIden;
    geomCode += Co && CoOvLi ? geomMain_PS_v_doColorOverLT : Co ? geomMain_PS_v_c
                                                                : geomMain_PS_v_withoutColor;
    geomCode += geomMain_PS_v_cT;
    if (billboardType == BT_Vertical)
        geomCode += FlBoTex ? geomMain_PS_Flipbook_fourCorners_vertBillboard : geomMain_PS_fourCorners_vertBillboard;
    else if (billboardType == BT_Horizontal)
        geomCode += FlBoTex ? geomMain_PS_Flipbook_fourCorners_horizBillboard : geomMain_PS_fourCorners_horizBillboard;
    else
        geomCode += FlBoTex ? geomMain_PS_Flipbook_fourCorners : geomMain_PS_fourCorners;

    geomCode += geomMain_PS_EndAll;

    addCodeToShader(_shaders[2], geomCode, _name + ".geom");

    ////////////////////////////////
    // Assemble fragment shader code
    ////////////////////////////////

    string fragCode;
    fragCode += shaderHeader();

    // Fragment shader inputs
    fragCode += fragInput_PS_v_pC;
    fragCode += fragInput_PS_v_tC;

    // Fragment shader uniforms
    if (Dm) fragCode += fragInput_u_matTexDm;
    fragCode += fragInput_PS_u_overG;
    fragCode += fragInput_PS_u_wireFrame;

    // Fragment shader outputs
    fragCode += fragOutputs_o_fragColor;

    // Fragment shader main loop
    fragCode += Co ? fragMain_PS : fragMain_PS_withoutColor;
    fragCode += fragMain_PS_endAll;

    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixParticleUpdate(SLMaterial* mat)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    bool counterGap = mat->ps()->doCounterGap();      // Counter gap/lag
    bool acc        = mat->ps()->doAcc();             // Acceleration
    bool accDiffDir = mat->ps()->doAccDiffDir();      // Acceleration different direction
    bool gravity    = mat->ps()->doGravity();         // Gravity
    bool FlBoTex    = mat->ps()->doFlipBookTexture(); // Flipbook texture
    bool rot        = mat->ps()->doRotation();        // Rotation
    bool rotRange   = mat->ps()->doRotRange();        // Rotation range
    bool shape      = mat->ps()->doShape();           // Shape

    string vertCode;
    vertCode += shaderHeader();

    // Vertex shader inputs
    vertCode += vertInput_PS_a_p;
    vertCode += vertInput_PS_a_v;
    vertCode += vertInput_PS_a_st;
    if (acc || gravity) vertCode += vertInput_PS_a_initV;
    if (rot) vertCode += vertInput_PS_a_r;
    if (rot && rotRange) vertCode += vertInput_PS_a_r_angularVelo;
    if (FlBoTex) vertCode += vertInput_PS_a_texNum;
    if (shape) vertCode += vertInput_PS_a_initP;

    // Vertex shader uniforms
    vertCode += vertInput_PS_u_time;
    vertCode += vertInput_PS_u_deltaTime;
    vertCode += vertInput_PS_u_pgPos;
    if (rot && !rotRange) vertCode += vertInput_PS_u_angularVelo;
    if (acc) vertCode += accDiffDir ? vertInput_PS_u_a_diffDir : vertInput_PS_u_a_const;
    if (gravity) vertCode += vertInput_PS_u_g;
    if (FlBoTex) vertCode += vertInput_PS_u_col;
    if (FlBoTex) vertCode += vertInput_PS_u_row;
    if (FlBoTex) vertCode += vertInput_PS_u_condFB;

    // Vertex shader outputs
    vertCode += vertOutput_PS_tf_p;
    vertCode += vertOutput_PS_tf_v;
    vertCode += vertOutput_PS_tf_st;
    if (acc || gravity) vertCode += vertOutput_PS_tf_initV;
    if (rot) vertCode += vertOutput_PS_tf_r;
    if (rot && rotRange) vertCode += vertOutput_PS_tf_r_angularVelo;
    if (FlBoTex) vertCode += vertOutput_PS_tf_texNum;
    if (shape) vertCode += vertOutput_PS_tf_initP;

    if (rot) vertCode += vertConstant_PS_pi; // Add constant PI

    // Vertex shader main loop
    vertCode += vertMain_PS_U_Begin;
    vertCode += vertMain_PS_U_v_init_p;
    vertCode += vertMain_PS_U_v_init_v;
    vertCode += vertMain_PS_U_v_init_st;
    if (acc || gravity) vertCode += vertMain_PS_U_v_init_initV;
    if (rot) vertCode += vertMain_PS_U_v_init_r;
    if (rot && rotRange) vertCode += vertMain_PS_U_v_init_r_angularVelo;
    if (FlBoTex) vertCode += vertMain_PS_U_v_init_texNum;
    if (shape) vertCode += vertMain_PS_U_v_init_initP;
    vertCode += vertMain_PS_U_bornDead;
    vertCode += shape ? vertMain_PS_U_reset_shape_p : vertMain_PS_U_reset_p;
    if (acc || gravity) vertCode += vertMain_PS_U_reset_v;
    vertCode += counterGap ? vertMain_PS_U_reset_st_counterGap : vertMain_PS_U_reset_st;
    vertCode += vertMain_PS_U_alive_p;
    if (rot) vertCode += rotRange ? vertMain_PS_U_v_rRange : vertMain_PS_U_v_rConst;
    if (FlBoTex) vertCode += vertMain_PS_U_alive_texNum;
    if (acc) vertCode += accDiffDir ? vertMain_PS_U_alive_a_diffDir : vertMain_PS_U_alive_a_const;
    if (gravity) vertCode += vertMain_PS_U_alive_g;
    vertCode += vertMain_PS_U_EndAll;

    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader();

    // Fragment shader inputs
    fragCode += fragMain_PS_TF;

    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixVideoBkgdSm(SLVLight* lights)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInput_a_pn;
    vertCode += vertInput_u_matrices_all;
    vertCode += vertOutput_v_P_VS;
    vertCode += vertOutput_v_P_WS;
    vertCode += vertOutput_v_N_VS;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
)";
    fragCode += fragInput_u_lightAll;
    fragCode += fragInput_u_lightSm(lights);
    fragCode += fragInput_u_cam;
    fragCode += fragInput_u_matAmbi;
    fragCode += fragInput_u_matTexDm;
    fragCode += fragInput_u_matGetsSm;
    fragCode += fragInput_u_shadowMaps(lights);
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragFunctionShadowTest(lights);
    fragCode += fragFunctionDoColoredShadows;
    fragCode += fragMainVideoBkgd;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//! Returns true if at least one of the light does shadow mapping
bool SLGLProgramGenerated::lightsDoShadowMapping(SLVLight* lights)
{
    for (auto light : *lights)
    {
        if (light->createsShadows())
            return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
string SLGLProgramGenerated::fragInput_u_lightSm(SLVLight* lights)
{
    string u_lightSm = R"(
uniform vec4        u_lightPosWS[NUM_LIGHTS];               // position of light in world space
uniform bool        u_lightCreatesShadows[NUM_LIGHTS];      // flag if light creates shadows
uniform int         u_lightNumCascades[NUM_LIGHTS];         // number of cascades for cascaded shadowmap
uniform bool        u_lightDoSmoothShadows[NUM_LIGHTS];     // flag if percentage-closer filtering is enabled
uniform int         u_lightSmoothShadowLevel[NUM_LIGHTS];   // radius of area to sample for PCF
uniform float       u_lightShadowMinBias[NUM_LIGHTS];       // min. shadow bias value at 0° to N
uniform float       u_lightShadowMaxBias[NUM_LIGHTS];       // min. shadow bias value at 90° to N
uniform bool        u_lightUsesCubemap[NUM_LIGHTS];         // flag if light has a cube shadow map
uniform bool        u_lightsDoColoredShadows;               // flag if shadows should be colored
)";
    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLLight* light = lights->at(i);
        if (light->createsShadows())
        {
            SLShadowMap* shadowMap = light->shadowMap();

            if (shadowMap->useCubemap())
            {
                u_lightSm += "uniform mat4        u_lightSpace_" + std::to_string(i) + "[6];\n";
            }
            else if (light->doCascadedShadows())
            {
                u_lightSm += "uniform mat4        u_lightSpace_" + std::to_string(i) + "[" + std::to_string(shadowMap->numCascades()) + "];\n";
            }
            else
            {
                u_lightSm += "uniform mat4        u_lightSpace_" + std::to_string(i) + ";\n";
            }
        }
    }
    return u_lightSm;
}
//-----------------------------------------------------------------------------
string SLGLProgramGenerated::fragInput_u_shadowMaps(SLVLight* lights)
{
    string smDecl = "\n";
    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLLight* light = lights->at(i);
        if (light->createsShadows())
        {
            SLShadowMap* shadowMap = light->shadowMap();
            if (shadowMap->useCubemap())
                smDecl += "uniform samplerCube u_shadowMapCube_" + to_string(i) + ";\n";
            else if (light->doCascadedShadows())
            {
                for (int j = 0; j < light->shadowMap()->depthBuffers().size(); j++)
                    smDecl += "uniform sampler2D   u_cascadedShadowMap_" + to_string(i) + "_" + std::to_string(j) + ";\n";

                smDecl += "uniform float       u_cascadesFactor_" + to_string(i) + ";\n";
            }
            else
                smDecl += "uniform sampler2D   u_shadowMap_" + to_string(i) + ";\n";
        }
    }
    return smDecl;
}
//-----------------------------------------------------------------------------
//! Adds the core shadow mapping test routine depending on the lights
string SLGLProgramGenerated::fragFunctionShadowTest(SLVLight* lights)
{
    bool doCascadedSM = false;
    for (SLLight* light : *lights)
    {
        if (light->doCascadedShadows())
        {
            doCascadedSM = true;
            break;
        }
    }

    string shadowTestCode = R"(
//-----------------------------------------------------------------------------
int vectorToFace(vec3 vec) // Vector to process
{
    vec3 absVec = abs(vec);
    if (absVec.x > absVec.y && absVec.x > absVec.z)
        return vec.x > 0.0 ? 0 : 1;
    else if (absVec.y > absVec.x && absVec.y > absVec.z)
        return vec.y > 0.0 ? 2 : 3;
    else
        return vec.z > 0.0 ? 4 : 5;
}
//-----------------------------------------------------------------------------
int getCascadesDepthIndex(in int i, int numCascades)
{
    float factor;
)";

    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLShadowMap* shadowMap = lights->at(i)->shadowMap();
        if (shadowMap && shadowMap->useCascaded())
        {
            shadowTestCode += "    if (i == " + std::to_string(i) + ")\n";
            shadowTestCode += "    {\n";
            shadowTestCode += "        factor = u_cascadesFactor_" + std::to_string(i) + ";\n";
            shadowTestCode += "    }\n";
        }
    }

    shadowTestCode += R"(
    float fi = u_camClipNear;
    float ni;

    for (int i = 0; i < numCascades-1; i++)
    {
        ni = fi;
        fi = factor * u_camClipNear * pow((u_camClipFar/(factor*u_camClipNear)), float(i+1)/float(numCascades));
        if (-v_P_VS.z < fi)
            return i;
    }
    return numCascades-1;
}
//-----------------------------------------------------------------------------
float shadowTest(in int i, in vec3 N, in vec3 lightDir)
{
    if (u_lightCreatesShadows[i])
    {
        // Calculate position in light space
        mat4 lightSpace;
        vec3 lightToFragment = v_P_WS - u_lightPosWS[i].xyz;
)";

    if (doCascadedSM > 0)
    {
        shadowTestCode += R"(
        int index = 0;

        if (u_lightNumCascades[i] > 0)
        {
            index = getCascadesDepthIndex(i, u_lightNumCascades[i]);
        )";
        for (SLuint i = 0; i < lights->size(); ++i)
        {
            SLShadowMap* shadowMap = lights->at(i)->shadowMap();
            if (shadowMap && shadowMap->useCascaded())
                shadowTestCode += "    if (i == " + std::to_string(i) + ") { lightSpace = u_lightSpace_" + std::to_string(i) + "[index]; }\n";
        }
        shadowTestCode += R"(
        }
        else if (u_lightUsesCubemap[i])
        {)";
        for (SLuint i = 0; i < lights->size(); ++i)
        {
            SLShadowMap* shadowMap = lights->at(i)->shadowMap();
            if (shadowMap && shadowMap->useCubemap())
                shadowTestCode += "    if (i == " + std::to_string(i) + ") { lightSpace = u_lightSpace_" + std::to_string(i) + "[vectorToFace(lightToFragment)]; }\n";
        }
        shadowTestCode += R"(
        }
        else
        {
        )";
        for (SLuint i = 0; i < lights->size(); ++i)
        {
            SLShadowMap* shadowMap = lights->at(i)->shadowMap();
            if (shadowMap && !shadowMap->useCubemap() && !shadowMap->useCascaded())
                shadowTestCode += "if (i == " + std::to_string(i) + ") { lightSpace = u_lightSpace_" + std::to_string(i) + "}\n";
        }
        shadowTestCode += R"(
        }
        )";
    }
    else
    {
        shadowTestCode += R"(
        if (u_lightUsesCubemap[i])
        {
)";
        for (SLuint i = 0; i < lights->size(); ++i)
        {
            SLShadowMap* shadowMap = lights->at(i)->shadowMap();
            if (shadowMap && shadowMap->useCubemap())
                shadowTestCode += "            if (i == " + std::to_string(i) + ") lightSpace = u_lightSpace_" + std::to_string(i) + "[vectorToFace(lightToFragment)];\n";
        }
        shadowTestCode += R"(
        }
        else
        {
)";
        for (SLuint i = 0; i < lights->size(); ++i)
        {
            SLShadowMap* shadowMap = lights->at(i)->shadowMap();
            if (shadowMap && !shadowMap->useCubemap() && !shadowMap->useCascaded())
                shadowTestCode += "            if (i == " + std::to_string(i) + ") lightSpace = u_lightSpace_" + std::to_string(i) + ";\n";
        }
        shadowTestCode += R"(
        }
        )";
    }

    shadowTestCode += R"(
        vec4 lightSpacePosition = lightSpace * vec4(v_P_WS, 1.0);

        // Normalize lightSpacePosition
        vec3 projCoords = lightSpacePosition.xyz / lightSpacePosition.w;

        // Convert to texture coordinates
        projCoords = projCoords * 0.5 + 0.5;

        float currentDepth = projCoords.z;

        // Look up depth from shadow map
        float shadow = 0.0;
        float closestDepth;

        // calculate bias between min. and max. bias depending on the angle between N and lightDir
        float bias = max(u_lightShadowMaxBias[i] * (1.0 - dot(N, lightDir)), u_lightShadowMinBias[i]);

        // Use percentage-closer filtering (PCF) for softer shadows (if enabled)
        if (u_lightDoSmoothShadows[i])
        {
            int level = u_lightSmoothShadowLevel[i];
            vec2 texelSize;
)";

    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLShadowMap* shadowMap = lights->at(i)->shadowMap();
        if (!shadowMap->useCascaded() && !shadowMap->useCubemap())
            shadowTestCode += "            if (i == " + to_string(i) + ") { texelSize = 1.0 / vec2(textureSize(u_shadowMap_" + to_string(i) + ", 0)); }\n";
        else if (shadowMap->useCascaded())
        {
            shadowTestCode += "            if (i == " + to_string(i) + ")\n            {\n";
            for (int j = 0; j < shadowMap->depthBuffers().size(); j++)
                shadowTestCode += "                if (index == " + to_string(j) + ") { texelSize = 1.0 / vec2(textureSize(u_cascadedShadowMap_" + to_string(i) + "_" + to_string(j) + ", 0)); }\n";

            shadowTestCode += "            }\n";
        }
    }
    shadowTestCode += R"(
            for (int x = -level; x <= level; ++x)
            {
                for (int y = -level; y <= level; ++y)
                {
    )";
    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLShadowMap* shadowMap = lights->at(i)->shadowMap();
        if (!shadowMap->useCascaded() && !shadowMap->useCubemap())
            shadowTestCode += "                 if (i == " + to_string(i) + ") { closestDepth = texture(u_shadowMap_" + to_string(i) + ", projCoords.xy + vec2(x, y) * texelSize).r; }\n";
        else if (shadowMap->useCascaded())
        {
            shadowTestCode += "                if (i == " + to_string(i) + ")\n                    {\n";
            for (int j = 0; j < shadowMap->depthBuffers().size(); j++)
                shadowTestCode += "                        if (index == " + to_string(j) + ") { closestDepth = texture(u_cascadedShadowMap_" + to_string(i) + "_" + to_string(j) + ", projCoords.xy + vec2(x, y) * texelSize).r; }\n";
            shadowTestCode += "                    }\n";
        }
    }

    shadowTestCode += R"(
                    shadow += currentDepth - bias > closestDepth ? 1.0 : 0.0;
                }
            }
            shadow /= pow(1.0 + 2.0 * float(level), 2.0);
        }
        else
        {
            if (u_lightUsesCubemap[i])
            {
)";
    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLShadowMap* shadowMap = lights->at(i)->shadowMap();
        if (shadowMap->useCubemap())
            shadowTestCode += "                if (i == " + to_string(i) + ") closestDepth = texture(u_shadowMapCube_" + to_string(i) + ", lightToFragment).r;\n";
    }
    shadowTestCode += R"(
            }
            else if (u_lightNumCascades[i] > 0)
            {
)";
    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLLight*     light     = lights->at(i);
        SLShadowMap* shadowMap = lights->at(i)->shadowMap();
        if (light->doCascadedShadows())
        {
            shadowTestCode += "                if (i == " + to_string(i) + ")\n                {\n";
            for (int j = 0; j < shadowMap->depthBuffers().size(); j++)
                shadowTestCode += "                    if (index == " + to_string(j) + ") { closestDepth = texture(u_cascadedShadowMap_" + to_string(i) + "_" + to_string(j) + ", projCoords.xy).r; }\n";
            shadowTestCode += "                }";
        }
    }

    shadowTestCode += R"(
            }
            else
            {
)";

    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLShadowMap* shadowMap = lights->at(i)->shadowMap();
        if (!shadowMap->useCubemap() && !shadowMap->useCascaded())
            shadowTestCode += "                if (i == " + to_string(i) + ") closestDepth = texture(u_shadowMap_" + to_string(i) + ", projCoords.xy).r;\n";
    }

    shadowTestCode += R"(
            }

            // The fragment is in shadow if the light doesn't "see" it
            if (currentDepth > closestDepth + bias)
                shadow = 1.0;
        }

        return shadow;
    }

    return 0.0;
}
)";

    return shadowTestCode;
}
//-----------------------------------------------------------------------------
//! Add vertex shader code to the SLGLShader instance
void SLGLProgramGenerated::addCodeToShader(SLGLShader*   shader,
                                           const string& code,
                                           const string& name)
{

#if defined(DEBUG) && defined(_DEBUG)
    shader->code(SLGLShader::removeComments(code));
#else
    shader->code(code);
#endif
    shader->name(name);

    // Check if generatedShaderPath folder exists
    generatedShaderPath = SLGLProgramManager::configPath + "generatedShaders/";
    if (!Utils::dirExists(SLGLProgramManager::configPath))
        SL_EXIT_MSG("SLGLProgramGenerated::addCodeToShader: SLGLProgramManager::configPath not existing");

    if (!Utils::dirExists(generatedShaderPath))
    {
        bool dirCreated = Utils::makeDir(generatedShaderPath);
        if (!dirCreated)
            SL_EXIT_MSG("SLGLProgramGenerated::addCodeToShader: Failed to created SLGLProgramManager::configPath/generatedShaders");
    }

    shader->file(generatedShaderPath + name);
}
//-----------------------------------------------------------------------------
//! Adds shader header code
string SLGLProgramGenerated::shaderHeader(int numLights)

{
    string header = "\nprecision highp float;\n";
    header += "\n#define NUM_LIGHTS " + to_string(numLights) + "\n";
    return header;
}

//-----------------------------------------------------------------------------
//! Adds shader header code
string SLGLProgramGenerated::shaderHeader()

{
    string header = "\nprecision highp float;\n";
    return header;
}
//-----------------------------------------------------------------------------
