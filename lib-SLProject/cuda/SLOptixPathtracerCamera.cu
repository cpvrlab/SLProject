#include <SLOptixDefinitions.h>
#include <SLOptixHelper.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__sample_camera()
{
    uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const CameraData* rtData = (CameraData*)optixGetSbtDataPointer();

    curandState state;
    curand_init(idx.y * dim.x + idx.x, 0,0, &state);
    params.states[idx.y * dim.x + idx.x] = state;

    float4 color = make_float4(0.0f);
    for (int i = 0; i < params.samples; i++) {
        const float2 subpixel_jitter = make_float2( curand_uniform(&state) - 0.5f, curand_uniform(&state) - 0.5f );

        const float2 pixel_offset = 2.0f * make_float2(
                ( static_cast<float>( idx.x ) + subpixel_jitter.x ) / static_cast<float>( dim.x ),
                ( static_cast<float>( idx.y ) + subpixel_jitter.y ) / static_cast<float>( dim.y )
        ) - 1.0f;

        // Calculate ray origin and direction
        const float3 origin      = rtData->eye;
        const float3 direction   = normalize( pixel_offset.x * rtData->U + pixel_offset.y * rtData->V + rtData->W );

        color += tracePrimaryRay(params.handle, origin, direction) / params.samples;
    }

    // Set pixel color
    params.image[idx.y * dim.x + idx.x] = make_color( color );
}