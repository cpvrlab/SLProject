#version 450

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragPos;
layout(location = 3) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
	float ambient = 0.3f;
	vec3 lightPos = vec3(15.0f, 15.0f, 15.0f);
	
	 vec3 norm = normalize(fragNormal);
	 vec3 lightDir = normalize(lightPos - fragPos);
	 float diff = max(dot(norm, lightDir), 0.0f);
	 vec3 diffuse = vec3(diff);
	 vec3 result = ambient + diffuse;
     outColor = vec4(result, 1.0f) * texture(texSampler, fragTexCoord) * fragColor; 
}