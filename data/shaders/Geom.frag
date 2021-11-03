precision highp float;

//-----------------------------------------------------------------------------
in vec4 v_particleColor;
in vec2 v_texCoord;          // interpolated texture coordinate

uniform sampler2D  u_matTextureDiffuse0;  // texture map

out vec4 FragColor;

void main()
{
    FragColor = v_particleColor; 

    FragColor *= texture(u_matTextureDiffuse0, v_texCoord);
}  