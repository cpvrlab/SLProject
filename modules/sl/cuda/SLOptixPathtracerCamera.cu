//#############################################################################
//  File:      SLOptixPathtracerCamera.cu
//  Purpose:   CUDA Shader file used in Optix Tracing
//  Date:      October 2019
//  Authors:   Nic Dorner, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLOptixDefinitions.h>
#include <SLOptixHelper.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

//-----------------------------------------------------------------------------
extern "C" {
__constant__ ortParams params;
}
//-----------------------------------------------------------------------------
extern "C" __global__ void __raygen__sample_camera()
{
    uint3       idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const ortCamera* rtData = (ortCamera*)optixGetSbtDataPointer();

    // Generate curand state
    curandState state;
    curand_init(idx.y * dim.x + idx.x, 0, 0, &state);
    params.states[idx.y * dim.x + idx.x] = state;

    float4 color = make_float4(0.0f);

    // loop over samples
    for (int i = 0; i < params.samples; i++)
    {
        // Random displacement
        const float2 subpixel_jitter = make_float2(curand_uniform(&state) - 0.5f,
                                                   curand_uniform(&state) - 0.5f);

        // Get pixel offset and add random displacement
        const float2 pixel_offset = 2.0f * make_float2(
                                             (static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(dim.x),
                                             (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(dim.y)) -
                                    1.0f;

        // Calculate ray origin and direction
        const float3 origin    = rtData->eye;
        const float3 direction = normalize(pixel_offset.x * rtData->U + pixel_offset.y * rtData->V + rtData->W);

        color += tracePrimaryRay(params.handle, origin, direction);
    }

    // Set pixel color
    params.image[idx.y * dim.x + idx.x] = gamma_correction(make_color(color / params.samples), 0.5f);
}
//-----------------------------------------------------------------------------