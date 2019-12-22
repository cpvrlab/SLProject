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

extern "C" __global__ void __anyhit__radiance() {
}

static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
    // Uniformly sample disk.
    const float r   = sqrtf( u1 );
    const float phi = 2.0f*M_PIf * u2;
    p.x = r * cosf( phi );
    p.y = r * sinf( phi );

    // Project up to hemisphere.
    p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) );
}

extern "C" __global__ void __closesthit__radiance() {
    // Get all data for the hit point
    auto *rt_data = reinterpret_cast<HitData *>( optixGetSbtDataPointer());
    const float3 ray_dir = optixGetWorldRayDirection();

    uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    curandState *state = &params.states[idx.y * dim.x + idx.x];

    // calculate normal vector
    float3 N = getNormalVector();
    // calculate texture color
    float4 texture_color = getTextureColor();

    // calculate hit point
    const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    if (rt_data->material.emissive_color.x != 0 ||
        rt_data->material.emissive_color.y != 0 ||
        rt_data->material.emissive_color.z != 0) {
        setColor(rt_data->material.emissive_color);
    } else {
        // initialize color
        float4 local_color = (texture_color * rt_data->material.diffuse_color);

        float3 direction;
        cosine_sample_hemisphere( curand_uniform(state), curand_uniform(state), direction );

        float4 incoming_color = make_float4(0.0f);
        if (getDepth() < params.max_depth) {
            incoming_color = traceSecondaryRay(params.handle, P, direction);
        }

        // Set color to payload
        setColor(local_color * incoming_color);
    }
}