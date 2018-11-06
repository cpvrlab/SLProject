
#version 120

varying vec2	v_texPos;

uniform sampler2D	u_texture0;
uniform sampler2D	u_texture1;


void main() {
	//if(v_texPos.x * v_texPos.y > 0.2) {
		gl_FragColor = texture2D(u_texture0, v_texPos);
	//} else {
	//	gl_FragColor = texture2D(u_texture1, v_texPos);
	//}
}