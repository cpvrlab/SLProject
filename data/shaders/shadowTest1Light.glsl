//! Shadow text function for 1 light without cubemap shadow maps
float shadowTest1Light(in int i, in vec3 N, in vec3 lightDir)
{
    if (u_lightCreatesShadows[i])
    {
        // Calculate position in light space
        vec3 lightToFragment = v_P_WS - u_lightPosWS[i].xyz;
        vec4 lightSpacePosition = u_lightSpace_0 * vec4(v_P_WS, 1.0);

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
            vec2 texelSize;
            texelSize = 1.0 / vec2(textureSize(u_shadowMap_0, 0));
            int level = u_lightSmoothShadowLevel[i];

            for (int x = -level; x <= level; ++x)
            {
                for (int y = -level; y <= level; ++y)
                {
                    closestDepth = texture(u_shadowMap_0, projCoords.xy + vec2(x, y) * texelSize).r;
                    shadow += currentDepth - bias > closestDepth ? 1.0 : 0.0;
                }
            }
            shadow /= pow(1.0 + 2.0 * float(level), 2.0);
        }
        else
        {
            closestDepth = texture(u_shadowMap_0, projCoords.xy).r;

            // The fragment is in shadow if the light doesn't "see" it
            if (currentDepth > closestDepth + bias)
                shadow = 1.0;
        }

        return shadow;
    }

    return 0.0;
}
