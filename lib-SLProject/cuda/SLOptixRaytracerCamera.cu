#include <SLOptixDefinitions.h>
#include <SLOptixRaytracerHelper.h>

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const CameraData* rtData = (CameraData*)optixGetSbtDataPointer();
    const float3      U      = rtData->U;
    const float3      V      = rtData->V;
    const float3      W      = rtData->W;
    const float2      d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
    ) - 1.0f;

    const float3 origin      = rtData->eye;
    const float3 direction   = normalize( d.x * U + d.y * V + W );

    params.image[idx.y * params.width + idx.x] = make_color( traceRadianceRay(params.handle, origin, direction, 1.0f, false, 1) );
}