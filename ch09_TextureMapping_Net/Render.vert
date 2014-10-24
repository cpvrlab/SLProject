#version 120


attribute vec4	a_position;
attribute vec2  a_texPos;

uniform mat4	u_mvpMatrix;

varying vec2 	v_texPos;

void main() {
	v_texPos = a_texPos;
	gl_Position = u_mvpMatrix * a_position;
}