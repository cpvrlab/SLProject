#version 430 core

// uniform mat4 u_vMatrix; this not needed?

layout(location = 0) in vec4 a_position;

out vec2 textureCoordinateFrag; 

// Scales and bias a given vector (i.e. from [-1, 1] to [0, 1]).
vec2 scaleAndBias(vec2 p) { return 0.5f * p + vec2(0.5f); }

void main(){
	textureCoordinateFrag = scaleAndBias(a_position.xy);
	gl_Position = a_position;
}
