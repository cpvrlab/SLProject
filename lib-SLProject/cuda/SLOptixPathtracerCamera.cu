#include <SLOptixDefinitions.h>
#include <SLOptixHelper.h>
#include <cuda_runtime_api.h>

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__sample_camera()
{
    uint3 launch_index = optixGetLaunchIndex();
    params.image[launch_index.y * params.width + launch_index.x] = make_uchar4( 255, 0, 0, 255 );
}