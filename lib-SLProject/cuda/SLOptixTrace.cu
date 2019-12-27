#include <SLOptixHelper.h>
#include <SLOptixDefinitions.h>
#include <cuda_runtime_api.h>

extern "C" __global__ void __intersection__line()
{
    auto *rt_data = reinterpret_cast<HitData *>( optixGetSbtDataPointer());
    const Line line = rt_data->geometry.line;

    const float3  ray_orig = optixGetWorldRayOrigin();
    const float3  ray_dir  = optixGetWorldRayDirection();

    const float3 line_orig = line.p1;
    const float3 line_dir  = line.p2 - line.p1;

    float u = (ray_orig.y * line_dir.x + line_dir.y * line_orig.x - line_orig.y * line_dir.x - line_dir.y * ray_orig.x ) / (ray_dir.x * line_dir.y - ray_dir.y * line_dir.x);
    float v = (ray_orig.x + ray_dir.x * u - line_orig.x) / line_dir.x;

    if (u >= 0.0f && v >= 0.0f && v <= 1.0f) {
        float3 p1 = ray_orig + ray_dir * u;
        float3 p2 = line_orig + line_dir * v;

        if (abs(length(p2 - p1)) <= 0.01f) {
            optixReportIntersection( length(p1 - ray_orig), 0);
        }
    }
}

extern "C" __global__ void __anyhit__line_radiance() {
}

extern "C" __global__ void __anyhit__line_occlusion() {
}

extern "C" __global__ void __closesthit__line_radiance() {
    auto *rt_data = reinterpret_cast<HitData *>( optixGetSbtDataPointer());

    // Set color to payload
    setColor(rt_data->material.diffuse_color);
}