#include <SLOptixDefinitions.h>
#include <SLOptixHelper.h>
#include <math_functions.h>

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__pinhole_camera()
{
    // Get ray generation data
    const uint3 idx = optixGetLaunchIndex();
    const CameraData* rtData = (CameraData*)optixGetSbtDataPointer();

    const float2 pixel_offset = getPixelOffset(idx);

    // Calculate ray origin and direction
    const float3 origin      = rtData->eye;
    const float3 direction   = normalize( pixel_offset.x * rtData->U + pixel_offset.y * rtData->V + rtData->W );

    // Set pixel color
    params.image[idx.y * params.width + idx.x] = make_color( tracePrimaryRay(params.handle, origin, direction) );
}