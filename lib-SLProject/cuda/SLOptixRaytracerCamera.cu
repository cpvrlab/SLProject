#include <SLOptixDefinitions.h>
#include <SLOptixRaytracerHelper.h>
#include <math_functions.h>

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__draw_solid_color()
{
    uint3 launch_index = optixGetLaunchIndex();
    params.image[launch_index.y * params.width + launch_index.x] = make_uchar4( 255, 0, 0, 255 );
}

extern "C" __global__ void __raygen__pinhole_camera()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const CameraData* rtData = (CameraData*)optixGetSbtDataPointer();
    const float2      d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
    ) - 1.0f;

    const float3 origin      = rtData->eye;
    const float3 direction   = normalize( d.x * rtData->U + d.y * rtData->V + rtData->W );

    params.image[idx.y * params.width + idx.x] = make_color( tracePrimaryRay(params.handle, origin, direction) );
}

extern "C" __global__ void __raygen__lens_camera()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const LensCameraData* rtData = (LensCameraData*)optixGetSbtDataPointer();
    const float2      d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
    ) - 1.0f;

    const float3 pixel_pos = d.x * rtData->camera.U + d.y * rtData->camera.V + rtData->camera.W + rtData->camera.eye;

    float radius = rtData->lensDiameter / 2.0f;
    float4 color = make_float4(0.0f);

    for (unsigned int r = 1; r <= rtData->samplesX; r++) {
        for (unsigned int q = 1; q <= rtData->samplesY; q++) {
            const float phi = (2.0f / rtData->samplesY) * q;
            const float3 origin      = rtData->camera.eye +
                    (normalize(rtData->camera.U) * cospif(phi) * ((radius / rtData->samplesX) * r)) +
                    (normalize(rtData->camera.V) * sinpif(phi) * ((radius / rtData->samplesX) * r));
            const float3 direction   = normalize(pixel_pos - origin);

            color += tracePrimaryRay(params.handle, origin, direction);
        }
    }


    params.image[idx.y * params.width + idx.x] = make_color( color / (rtData->samplesX * rtData->samplesY) );
}


extern "C" __global__ void __raygen__orthographic_camera()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const CameraData* rtData = (CameraData*)optixGetSbtDataPointer();
    const float2      d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
    ) - 1.0f;

    const float3 origin     = d.x * rtData->U + d.y * rtData->V + rtData->eye;;
    const float3 direction  = normalize(rtData->W);

    params.image[idx.y * params.width + idx.x] = make_color( tracePrimaryRay(params.handle, origin, direction) );
}