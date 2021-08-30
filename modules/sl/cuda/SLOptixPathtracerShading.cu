//#############################################################################
//  File:      SLOptixPathtracerShading.cu
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
extern "C" __global__ void __miss__sample()
{
    auto* rt_data = reinterpret_cast<ortMissData*>(optixGetSbtDataPointer());
    setColor(rt_data->bg_color);
}
//-----------------------------------------------------------------------------
extern "C" __global__ void __miss__occlusion()
{
}
//-----------------------------------------------------------------------------
extern "C" __global__ void __anyhit__radiance()
{
}
//-----------------------------------------------------------------------------
extern "C" __global__ void __closesthit__radiance()
{
    // Get all data for the hit point
    auto*        rt_data = reinterpret_cast<ortHitData*>(optixGetSbtDataPointer());
    const float3 ray_dir = optixGetWorldRayDirection();

    if (getDepth() > params.max_depth)
    {
        setColor(make_float4(0.0f));
        return;
    }

    uint3        idx   = optixGetLaunchIndex();
    const uint3  dim   = optixGetLaunchDimensions();
    curandState* state = &params.states[idx.y * dim.x + idx.x];

    // calculate normal vector
    float3 N = getNormalVector();

    // calculate texture color
    float4 texture_color = getTextureColor();

    // calculate hit point
    const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    if (rt_data->material.emissive_color.x != 0 ||
        rt_data->material.emissive_color.y != 0 ||
        rt_data->material.emissive_color.z != 0)
    {
        setColor(rt_data->material.emissive_color);
    }
    else
    {
        // initialize color
        float4 local_color;
        float4 incoming_color;
        float  random = curand_uniform(state);
        if (rt_data->material.kr > random)
        {
            incoming_color = traceReflectionRay(params.handle,
                                                P,
                                                N,
                                                ray_dir);
            local_color    = rt_data->material.specular_color;
        }
        else if ((rt_data->material.kr + rt_data->material.kt) > random)
        {
            incoming_color = traceRefractionRay(params.handle,
                                                P,
                                                N,
                                                ray_dir,
                                                rt_data->material.kn);
            local_color    = rt_data->material.transmissiv_color;
        }
        else
        {
            float3 direction;
            cosine_sample_hemisphere(curand_uniform(state),
                                     curand_uniform(state),
                                     N,
                                     direction);

            incoming_color = traceSecondaryRay(params.handle,
                                               P,
                                               direction);

            local_color    = rt_data->material.diffuse_color * texture_color;
        }

        // Set color to payload
        setColor(local_color * incoming_color);
    }
}
//-----------------------------------------------------------------------------