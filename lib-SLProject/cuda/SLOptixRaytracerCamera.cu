#include <optix.h>
#include <optix_device.h>
#include <optix_types.h>

#include <cuda_runtime.h>
#include <SLOptixDefinitions.h>

extern "C" {
__constant__ Params params;
}

extern "C"
__global__ void __raygen__draw_solid_color()
{
    optixThrowException(1);
    uint3 launch_index = optixGetLaunchIndex();
    CameraData* rtData = (CameraData*)optixGetSbtDataPointer();
    params.image[launch_index.y * params.image_width + launch_index.x] = make_uchar3( 255, 0, 0 );
}