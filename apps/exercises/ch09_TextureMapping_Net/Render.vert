#version 120

attribute vec4  a_position;
attribute vec2  a_texPos;

uniform mat4    u_mMatrix;        // Model matrix (object to world transform)
uniform mat4    u_vMatrix;        // View matrix (world to camera transform)
uniform mat4    u_pMatrix;        // Projection matrix (camera to normalize device coords.)

varying vec2    v_texPos;

void main()
{
    v_texPos = a_texPos;
    gl_Position = u_mvpMatrix * a_position;
}