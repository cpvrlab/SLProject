#version 430 core

in vec3 a_P_WS;

out vec4 color;

void main(){
  color.rgb = a_P_WS;
}
