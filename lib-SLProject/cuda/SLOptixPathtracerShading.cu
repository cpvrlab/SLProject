#include <SLOptixHelper.h>
#include <SLOptixDefinitions.h>
#include <cuda_runtime_api.h>

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __miss__sample() {
    auto *rt_data = reinterpret_cast<MissData *>( optixGetSbtDataPointer());
    setColor(rt_data->bg_color);
}