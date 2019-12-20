#include <SLOptixDefinitions.h>
#include <SLOptixHelper.h>
#include <cuda_runtime_api.h>

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

extern "C" __global__ void __raygen__lens_camera()
{
    const uint3 idx = optixGetLaunchIndex();

    const LensCameraData* rtData = (LensCameraData*)optixGetSbtDataPointer();
    const float2 pixel_offset = getPixelOffset(idx);

    const float3 pixel_pos = pixel_offset.x * rtData->camera.U + pixel_offset.y * rtData->camera.V + rtData->camera.W + rtData->camera.eye;

    float radius = rtData->lensDiameter / 2.0f;
    float4 color = make_float4(0.0f);

    for (unsigned int r = 1; r <= rtData->samples.samplesX; r++) {
        for (unsigned int q = 1; q <= rtData->samples.samplesY; q++) {
            const float phi = (2.0f / rtData->samples.samplesY) * q;
            const float3 origin      = rtData->camera.eye +
                    (normalize(rtData->camera.U) * cospif(phi) * ((radius / rtData->samples.samplesX) * r)) +
                    (normalize(rtData->camera.V) * sinpif(phi) * ((radius / rtData->samples.samplesX) * r));
            const float3 direction   = normalize(pixel_pos - origin);

            color += tracePrimaryRay(params.handle, origin, direction);
        }
    }


    params.image[idx.y * params.width + idx.x] = make_color( color / (rtData->samples.samplesX * rtData->samples.samplesY) );
}


extern "C" __global__ void __raygen__orthographic_camera()
{
    const uint3 idx = optixGetLaunchIndex();

    const CameraData* rtData = (CameraData*)optixGetSbtDataPointer();
    const float2 pixel_offset = getPixelOffset(idx);

    const float3 origin     = pixel_offset.x * rtData->U + pixel_offset.y * rtData->V + rtData->eye;;
    const float3 direction  = normalize(rtData->W);

    params.image[idx.y * params.width + idx.x] = make_color( tracePrimaryRay(params.handle, origin, direction) );
}