#include <SLOptixRaytracerHelper.h>
#include <SLOptixDefinitions.h>

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __miss__radiance()
{
    auto* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    setColor(rt_data->bg_color);
}

extern "C" __global__ void __miss__occlusion()
{
}

extern "C" __global__ void __anyhit__radiance()
{
}

extern "C" __global__ void __anyhit__occlusion()
{
    auto *rt_data = reinterpret_cast<HitData *>( optixGetSbtDataPointer());
    setLighted(getLighted() - (1.0f - rt_data->material.kt));
    optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__radiance() {
    // Get all data for the hit point
    auto *rt_data = reinterpret_cast<HitData *>( optixGetSbtDataPointer());
    unsigned int idx = optixGetPrimitiveIndex();
    const float3 ray_dir = optixGetWorldRayDirection();

    // calculate normal vector
    float3 N;
    float4 texture_color = make_float4(1.0);
    {
        const float2 barycentricCoordinates = optixGetTriangleBarycentrics();
        const float u = barycentricCoordinates.x;
        const float v = barycentricCoordinates.y;
        if (rt_data->normals && rt_data->indices) {
            // Interpolate normal vector with barycentric coordinates
            N = (1.f-u-v) * rt_data->normals[rt_data->indices[idx].x]
                +         u * rt_data->normals[rt_data->indices[idx].y]
                +         v * rt_data->normals[rt_data->indices[idx].z];
            N = normalize( optixTransformNormalFromObjectToWorldSpace( N ) );
        } else {
            OptixTraversableHandle gas = optixGetGASTraversableHandle();
            float3 vertex[3] = { make_float3(0.0f), make_float3(0.0f), make_float3(0.0f)};
            optixGetTriangleVertexData(gas,
                                       idx,
                                       rt_data->sbtIndex,
                                       0,
                                       vertex);
            N = normalize(cross(vertex[1] - vertex[0], vertex[2] - vertex[0]));
        }

        if (rt_data->textureObject) {
            const float2 tc
                    = (1.f-u-v) * rt_data->texCords[rt_data->indices[idx].x]
                      +         u * rt_data->texCords[rt_data->indices[idx].y]
                      +         v * rt_data->texCords[rt_data->indices[idx].z];
            texture_color = tex2D<float4>(rt_data->textureObject, tc.x, tc.y);
        }
    }

    // if a back face was hit then the normal vector is in the opposite direction
    if (optixIsTriangleBackFaceHit()) {
        N = N * -1;
    }
    // calculate hit point
    const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    // initialize color
    float4 color = make_float4(0.0f);
    {
        float4 local_color      = make_float4(0.0f);
        float4 specular_color   = make_float4(0.0f);;

        // Add emissive and ambient to current color
        local_color += rt_data->material.emissive_color;
        local_color += rt_data->material.ambient_color * params.globalAmbientColor;

        // calculate local illumination for every light source
        for (int i = 0; i < params.numLights; i++) {
            const Light light = params.lights[i];
            const float Ldist = length(light.position - P);
            const float3 L = normalize(light.position - P);
            const float nDl = dot(L, N);

            // Phong specular reflection
//        const float3 R = normalize(reflect(-L, N));
//        const float3 V = normalize(-ray_dir);
//        powf( max(dot(R, V), 0.0), rt_data->material.shininess )
            // Blinn specular reflection
            const float3 H = normalize(L - ray_dir); // half vector between light & eye

            if ( nDl > 0.0f)
            {
                uint32_t p0 = float_as_int( 1.0f );
                // Send shadow ray
                optixTrace(
                        params.handle,
                        P,
                        L,
                        1e-3f,                         // tmin
                        Ldist,                // tmax
                        0.0f,                       // rayTime
                        OptixVisibilityMask( 1 ),
                        OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                        RAY_TYPE_OCCLUSION,        // SBT offset
                        RAY_TYPE_COUNT,            // SBT stride
                        RAY_TYPE_OCCLUSION,     // missSBTIndex
                        p0 // payload
                );
                float lighted = min(int_as_float( p0 ), 1.0f);

                // Phong shading
                if (lighted > 0) {
                    // calculate spot effect if light is a spotlight
                    float spotEffect = 1.0f;;
                    if (lighted > 0.0f && light.spotCutOffDEG < 180.0f)
                    {
                        float LdS = max(dot(-L, light.spotDirWS), 0.0f);

                        // check if point is in spot cone
                        if (LdS > light.spotCosCut)
                        {
                            spotEffect = powf(LdS, light.spotExponent);
                        }
                        else
                        {
                            lighted    = 0.0f;
                            spotEffect = 0.0f;
                        }
                    }

                    local_color += (rt_data->material.diffuse_color * max(nDl, 0.0f))                                               // diffuse
                                   * lighted                                                                                              // lighted
                                   * light.diffuse_color                                                                       // multiply with diffuse light color
                                   * lightAttenuation(light, Ldist)                                                             // multiply with light attenuation
                                   * spotEffect;
                    specular_color += (rt_data->material.specular_color * powf( max(dot(N, H), 0.0), rt_data->material.shininess))  // specular
                                      * lighted                                                                                              // lighted
                                      * light.specular_color                                                                      // multiply with specular light color
                                      * lightAttenuation(light, Ldist)                                                          // multiply with light attenuation
                                      * spotEffect;
                }
            }
            local_color += rt_data->material.ambient_color
                    * lightAttenuation(light, Ldist)
                    * light.ambient_color;
        }

        // multiply local color with texture color and add specular color afterwards
        color += (local_color * texture_color) + specular_color;
    }

    // Send reflection ray
    if(getDepth() < params.max_depth && rt_data->material.kr > 0.0f) {
        float3 R = reflect(ray_dir, N);
        color += (traceReflectionRay(params.handle, P, R, getDepth() + 1) * rt_data->material.kr);
    }

    // Send refraction ray
    if(getDepth() < params.max_depth && rt_data->material.kt > 0.0f) {
        // calculate eta
        float refractionIndex = rt_data->material.kn;
        float eta = getRefractionIndex() / rt_data->material.kn;
        if (optixIsTriangleBackFaceHit()) {
            refractionIndex = 1.0f;
            eta = rt_data->material.kn / 1.0f;
        }

        // calculate transmission vector T
        float3 T;
        float c1 = dot(N, -ray_dir);
        float w = eta * c1;
        float c2 = 1.0f + (w - eta) * (w + eta);
        if(c2 >= 0.0f) {
            T = eta * ray_dir + (w - sqrtf(c2)) * N;
        } else {
            T = 2.0f * (dot(-ray_dir, N)) * N  + ray_dir;
        }
        color += (traceRefractionRay(params.handle, P, T, refractionIndex, getDepth() + 1) * rt_data->material.kt);
    }

    // Set color to payload
    setColor(color);
}