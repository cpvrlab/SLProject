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

    curandState_t state;
    curand_init(params.seed + idx.y * dim.x + idx.x, 0,0, &state);

    for (int i = 0; i < params.samples; i++) {
        const float2 subpixel_jitter = make_float2( curand_uniform(&state) - 0.5f, curand_uniform(&state) - 0.5f );

        const float2 pixel_offset = 2.0f * make_float2(
                ( static_cast<float>( idx.x ) + subpixel_jitter.x ) / static_cast<float>( dim.x ),
                ( static_cast<float>( idx.y ) + subpixel_jitter.y ) / static_cast<float>( dim.y )
        ) - 1.0f;

        // Calculate ray origin and direction
        const float3 origin      = rtData->eye;
        const float3 direction   = normalize( pixel_offset.x * rtData->U + pixel_offset.y * rtData->V + rtData->W );

        // Set pixel color
        params.image[idx.y * params.width + idx.x] = make_color( tracePrimaryRay(params.handle, origin, direction) );
    }
}