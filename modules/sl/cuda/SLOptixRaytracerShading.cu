//#############################################################################
//  File:      SLOptixRaytracerShading.cu
//  Purpose:   CUDA Shader file used in Optix Tracing
//  Date:      October 2019
//  Authors:   Nic Dorner, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLOptixHelper.h>
#include <SLOptixDefinitions.h>
#include <cuda_runtime_api.h>

//-----------------------------------------------------------------------------
extern "C" {
__constant__ ortParams params;
}
//-----------------------------------------------------------------------------
extern "C" __global__ void __miss__radiance()
{
    auto* rt_data = reinterpret_cast<ortMissData*>(optixGetSbtDataPointer());
    setColor(rt_data->bg_color);
}
//-----------------------------------------------------------------------------
extern "C" __global__ void __miss__occlusion()
{
}

extern "C" __global__ void __anyhit__radiance()
{
}
//-----------------------------------------------------------------------------
extern "C" __global__ void __anyhit__occlusion()
{
    auto* rt_data = reinterpret_cast<ortHitData*>(optixGetSbtDataPointer());
    setLighted(getLighted() - (1.0f - rt_data->material.kt));
    optixIgnoreIntersection();
}
//-----------------------------------------------------------------------------
extern "C" __global__ void __closesthit__radiance()
{
    uint3       idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Get all data for the hit point
    auto*        rt_data = reinterpret_cast<ortHitData*>(optixGetSbtDataPointer());
    const float3 ray_dir = optixGetWorldRayDirection();

    // calculate normal vector
    float3 N = getNormalVector();

    // calculate texture color
    float4 texture_color = getTextureColor();

    // calculate hit point
    const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    // initialize color
    float4 color = make_float4(0.0f);
    {
        float4 local_color    = make_float4(0.0f);
        float4 specular_color = make_float4(0.0f);

        // Add emissive and ambient to current color
        local_color += rt_data->material.emissive_color;
        local_color += rt_data->material.ambient_color * params.globalAmbientColor;

        // calculate local illumination for every light source
        for (int i = 0; i < params.numLights; i++)
        {
            const ortLight light = params.lights[i];
            const float    Ldist = length(light.position - P);
            const float3   L     = normalize(light.position - P);
            const float    LdotN = dot(L, N);

            // Phong specular reflection
            // const float3 R = normalize(reflect(-L, N));
            // const float3 V = normalize(-ray_dir);
            // powf( max(dot(R, V), 0.0), rt_data->material.shininess )

            // Blinn specular reflection
            const float3 H = normalize(L - ray_dir); // half vector between light & eye

            if (LdotN > 0.0f)
            {
                float lighted = 0.0f;

                float3 lightDiscX = normalize(make_float3(1, 1, (-1 * (L.x + L.y) / L.z)));
                float3 lightDiscY = cross(L, lightDiscX);

                for (unsigned int r = 1; r <= light.samples.samplesX; r++)
                {
                    for (unsigned int q = 1; q <= light.samples.samplesY; q++)
                    {
                        const float  phi       = (2.0f / light.samples.samplesY) * q;
                        const float3 discPoint = light.position +
                                                 (normalize(lightDiscX) * cospif(phi) * (light.radius / light.samples.samplesX) * r) +
                                                 (normalize(lightDiscY) * sinpif(phi) * (light.radius / light.samples.samplesX) * r);
                        const float3 direction = normalize(discPoint - P);
                        const float  discDist  = length(discPoint - P);

                        const float sampleLighted = traceShadowRay(params.handle,
                                                                   P,
                                                                   direction,
                                                                   discDist);
                        if (sampleLighted > 0)
                        {
                            lighted += sampleLighted / (light.samples.samplesX * light.samples.samplesY);
                        }
                    }
                }

                // Phong shading
                if (lighted > 0.0f)
                {
                    // calculate spot effect if light is a spotlight
                    float spotEffect = 1.0f;
                    ;
                    if (light.spotCutOffDEG < 180.0f)
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

                    local_color +=
                      (rt_data->material.diffuse_color * max(LdotN, 0.0f)) // diffuse
                      * lighted                                            // lighted
                      * light.diffuse_color                                // multiply with diffuse light color
                      * lightAttenuation(light, Ldist)                     // multiply with light attenuation
                      * spotEffect;

                    specular_color +=
                      (rt_data->material.specular_color * powf(max(dot(N, H), 0.0), rt_data->material.shininess)) // specular
                      * lighted                                                                                   // lighted
                      * light.specular_color                                                                      // multiply with specular light color
                      * lightAttenuation(light, Ldist)                                                            // multiply with light attenuation
                      * spotEffect;
                }
            }
            local_color += rt_data->material.ambient_color * lightAttenuation(light, Ldist) * light.ambient_color;
        }

        // multiply local color with texture color and add specular color afterwards
        color += (local_color * texture_color) + specular_color;
    }

    // Send reflection ray
    if (getDepth() < params.max_depth && rt_data->material.kr > 0.0f)
    {
        color += (traceReflectionRay(params.handle,
                                     P,
                                     N,
                                     ray_dir) *
                  rt_data->material.kr);
    }

    // Send refraction ray
    if (getDepth() < params.max_depth && rt_data->material.kt > 0.0f)
    {
        color += (traceRefractionRay(params.handle,
                                     P,
                                     N,
                                     ray_dir,
                                     rt_data->material.kn) *
                  rt_data->material.kt);
    }

    // Set color to payload
    setColor(color);
}
//-----------------------------------------------------------------------------