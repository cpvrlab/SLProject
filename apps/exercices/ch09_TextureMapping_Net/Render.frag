#version 120

in      vec2    v_texPos;

uniform sampler2D    u_matTexture0;
uniform sampler2D    u_matTexture1;


void main() {
    //if(v_texPos.x * v_texPos.y > 0.2) {
    gl_FragColor = texture2D(u_matTexture0, v_texPos);
    //} else {
    //	gl_FragColor = texture2D(u_matTexture1, v_texPos);
    //}
}