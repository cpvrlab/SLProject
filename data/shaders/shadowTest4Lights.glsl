//! Shadow text function for upto 4 lights without cubemap shadow maps
float shadowTest4Lights(in int i, in vec3 N, in vec3 lightDir)
{
    if (u_lightCreatesShadows[i])
    {
        // Calculate position in light space
        mat4 lightSpace;
        vec3 lightToFragment = v_P_WS - u_lightPosWS[i].xyz;
        lightSpace = u_lightSpace[i * 6];

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
            vec2 texelSize;
            if (i == 0) texelSize = 1.0 / vec2(textureSize(u_shadowMap_0, 0));
            if (i == 1) texelSize = 1.0 / vec2(textureSize(u_shadowMap_1, 0));
            if (i == 2) texelSize = 1.0 / vec2(textureSize(u_shadowMap_2, 0));
            if (i == 3) texelSize = 1.0 / vec2(textureSize(u_shadowMap_3, 0));
            int level = u_lightSmoothShadowLevel[i];

            for (int x = -level; x <= level; ++x)
            {
                for (int y = -level; y <= level; ++y)
                {
                    if (i == 0) closestDepth = texture(u_shadowMap_0, projCoords.xy + vec2(x, y) * texelSize).r;
                    if (i == 1) closestDepth = texture(u_shadowMap_1, projCoords.xy + vec2(x, y) * texelSize).r;
                    if (i == 2) closestDepth = texture(u_shadowMap_2, projCoords.xy + vec2(x, y) * texelSize).r;
                    if (i == 3) closestDepth = texture(u_shadowMap_3, projCoords.xy + vec2(x, y) * texelSize).r;
                    shadow += currentDepth - bias > closestDepth ? 1.0 : 0.0;
                }
            }
            shadow /= pow(1.0 + 2.0 * float(level), 2.0);
        }
        else
        {
            if (i == 0) closestDepth = texture(u_shadowMap_0, projCoords.xy).r;
            if (i == 1) closestDepth = texture(u_shadowMap_1, projCoords.xy).r;
            if (i == 2) closestDepth = texture(u_shadowMap_2, projCoords.xy).r;
            if (i == 3) closestDepth = texture(u_shadowMap_3, projCoords.xy).r;

            // The fragment is in shadow if the light doesn't "see" it
            if (currentDepth > closestDepth + bias)
                shadow = 1.0;
        }

        return shadow;
    }

    return 0.0;
}
