int vectorToFace(vec3 vec) // Vector to process
{
    vec3 absVec = abs(vec);
    if (absVec.x > absVec.y && absVec.x > absVec.z)
        return vec.x > 0.0 ? 0 : 1;
    else if (absVec.y > absVec.x && absVec.y > absVec.z)
        return vec.y > 0.0 ? 2 : 3;
    else
        return vec.z > 0.0 ? 4 : 5;
}

float shadowTest(in int i) // Light number
{
    if (u_lightCreatesShadows[i])
    {
        // Calculate position in light space
        mat4 lightSpace;
        vec3 lightToFragment = v_P_WS - u_lightPosWS[i].xyz;

        if (u_lightUsesCubemap[i])
            lightSpace = u_lightSpace[i * 6 + vectorToFace(lightToFragment)];
        else
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

        // Use percentage-closer filtering (PCF) for softer shadows (if enabled)
        if (!u_lightUsesCubemap[i] && u_lightDoSmoothShadows[i])
        {
            vec2 texelSize;
            if (i == 0) texelSize = 1.0 / vec2(textureSize(u_shadowMap_0, 0));
            if (i == 1) texelSize = 1.0 / vec2(textureSize(u_shadowMap_1, 0));
            if (i == 2) texelSize = 1.0 / vec2(textureSize(u_shadowMap_2, 0));
            if (i == 3) texelSize = 1.0 / vec2(textureSize(u_shadowMap_3, 0));
            if (i == 4) texelSize = 1.0 / vec2(textureSize(u_shadowMap_4, 0));
            if (i == 5) texelSize = 1.0 / vec2(textureSize(u_shadowMap_5, 0));
            if (i == 6) texelSize = 1.0 / vec2(textureSize(u_shadowMap_6, 0));
            if (i == 7) texelSize = 1.0 / vec2(textureSize(u_shadowMap_7, 0));
            int level = u_lightSmoothShadowLevel[i];

            for (int x = -level; x <= level; ++x)
            {
                for (int y = -level; y <= level; ++y)
                {
                    if (i == 0) closestDepth = texture(u_shadowMap_0, projCoords.xy + vec2(x, y) * texelSize).r;
                    if (i == 1) closestDepth = texture(u_shadowMap_1, projCoords.xy + vec2(x, y) * texelSize).r;
                    if (i == 2) closestDepth = texture(u_shadowMap_2, projCoords.xy + vec2(x, y) * texelSize).r;
                    if (i == 3) closestDepth = texture(u_shadowMap_3, projCoords.xy + vec2(x, y) * texelSize).r;
                    if (i == 4) closestDepth = texture(u_shadowMap_4, projCoords.xy + vec2(x, y) * texelSize).r;
                    if (i == 5) closestDepth = texture(u_shadowMap_5, projCoords.xy + vec2(x, y) * texelSize).r;
                    if (i == 6) closestDepth = texture(u_shadowMap_6, projCoords.xy + vec2(x, y) * texelSize).r;
                    if (i == 7) closestDepth = texture(u_shadowMap_7, projCoords.xy + vec2(x, y) * texelSize).r;
                    shadow += currentDepth - u_matShadowBias > closestDepth ? 1.0 : 0.0;
                }
            }
            shadow /= pow(1.0 + 2.0 * float(level), 2.0);
        }
        else
        {
            if (u_lightUsesCubemap[i])
            {
                if (i == 0) closestDepth = texture(u_shadowMapCube_0, lightToFragment).r;
                if (i == 1) closestDepth = texture(u_shadowMapCube_1, lightToFragment).r;
                if (i == 2) closestDepth = texture(u_shadowMapCube_2, lightToFragment).r;
                if (i == 3) closestDepth = texture(u_shadowMapCube_3, lightToFragment).r;
                if (i == 4) closestDepth = texture(u_shadowMapCube_4, lightToFragment).r;
                if (i == 5) closestDepth = texture(u_shadowMapCube_5, lightToFragment).r;
                if (i == 6) closestDepth = texture(u_shadowMapCube_6, lightToFragment).r;
                if (i == 7) closestDepth = texture(u_shadowMapCube_7, lightToFragment).r;
            }
            else
            {
                if (i == 0) closestDepth = texture(u_shadowMap_0, projCoords.xy).r;
                if (i == 1) closestDepth = texture(u_shadowMap_1, projCoords.xy).r;
                if (i == 2) closestDepth = texture(u_shadowMap_2, projCoords.xy).r;
                if (i == 3) closestDepth = texture(u_shadowMap_3, projCoords.xy).r;
                if (i == 4) closestDepth = texture(u_shadowMap_4, projCoords.xy).r;
                if (i == 5) closestDepth = texture(u_shadowMap_5, projCoords.xy).r;
                if (i == 6) closestDepth = texture(u_shadowMap_6, projCoords.xy).r;
                if (i == 7) closestDepth = texture(u_shadowMap_7, projCoords.xy).r;
            }

            // The fragment is in shadow if the light doesn't "see" it
            if (currentDepth > closestDepth + u_matShadowBias)
                shadow = 1.0;
        }

        return shadow;
    }

    return 0.0;
}

