#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>
#include <optix_types.h>
#include <SLOptixDefinitions.h>
#include <SLOptixVectorMath.h>

static __forceinline__ __device__ void setColor(float4 p) {
    optixSetPayload_0(float_as_int(p.x));
    optixSetPayload_1(float_as_int(p.y));
    optixSetPayload_2(float_as_int(p.z));
    optixSetPayload_3(float_as_int(p.w));
}

static __forceinline__ __device__ float getRefractionIndex() {
    return int_as_float(optixGetPayload_4());
}

static __forceinline__ __device__ unsigned int getDepth() {
    return optixGetPayload_5();
}

static __forceinline__ __device__ void setLighted(float lighted) {
    optixSetPayload_0(float_as_int(lighted));
}

static __forceinline__ __device__ float getLighted() {
    return int_as_float(optixGetPayload_0());
}

static __forceinline__ __device__ float lightAttenuation(Light light, float dist) {
    return 1.0f / (light.kc + light.kl * dist + light.kq * dist * dist);
}

__forceinline__ __device__ uchar4 make_color(const float4 &c) {
    return make_uchar4(
            static_cast<uint8_t>( clamp(c.x, 0.0f, 1.0f) * 255.0f ),
            static_cast<uint8_t>( clamp(c.y, 0.0f, 1.0f) * 255.0f ),
            static_cast<uint8_t>( clamp(c.z, 0.0f, 1.0f) * 255.0f ),
            static_cast<uint8_t>( clamp(c.w, 0.0f, 1.0f) * 255.0f )
    );
}

static __device__ __inline__ float4 tracePrimaryRay(
        OptixTraversableHandle handle,
        float3 origin,
        float3 direction) {
    float4 payload_rgb = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
    uint32_t p0, p1, p2, p3, p4, p5;
    p0 = float_as_int(payload_rgb.x);
    p1 = float_as_int(payload_rgb.y);
    p2 = float_as_int(payload_rgb.z);
    p3 = float_as_int(payload_rgb.w);
    p4 = float_as_int(1.0f);
    p5 = 1;
    optixTrace(
            handle,
            origin,
            direction,
            1.e-4f,  // tmin
            1e16f,  // tmax
            0.0f,                // rayTime
            OptixVisibilityMask( 255 ),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            RAY_TYPE_RADIANCE,                   // SBT offset
            RAY_TYPE_COUNT,                   // SBT stride
            RAY_TYPE_RADIANCE,                   // missSBTIndex
            p0, p1, p2, p3, p4, p5);
    payload_rgb.x = int_as_float(p0);
    payload_rgb.y = int_as_float(p1);
    payload_rgb.z = int_as_float(p2);
    payload_rgb.w = int_as_float(p3);

    return payload_rgb;
}

static __device__ __inline__ float traceShadowRay(
        OptixTraversableHandle handle,
        float3 origin,
        float3 direction,
        float dist) {
    uint32_t p0 = float_as_int( 1.0f );
    // Send shadow ray
    optixTrace(
            handle,
            origin,
            direction,
            1e-3f,                         // tmin
            dist,                // tmax
            0.0f,                       // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            RAY_TYPE_OCCLUSION,        // SBT offset
            RAY_TYPE_COUNT,            // SBT stride
            RAY_TYPE_OCCLUSION,     // missSBTIndex
            p0 // payload
    );
    float lighted = min(int_as_float( p0 ), 1.0f);

    return lighted;
}

static __device__ __inline__ float4 traceReflectionRay(
        OptixTraversableHandle handle,
        float3 origin,
        float3 direction,
        unsigned int depth) {
    float4 payload_rgb = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
    uint32_t p0, p1, p2, p3, p4;
    p0 = float_as_int(payload_rgb.x);
    p1 = float_as_int(payload_rgb.y);
    p2 = float_as_int(payload_rgb.z);
    p3 = float_as_int(payload_rgb.w);
    p4 = float_as_int(getRefractionIndex());
    optixTrace(
            handle,
            origin,
            direction,
            1.e-4f,  // tmin
            1e16f,  // tmax
            0.0f,                // rayTime
            OptixVisibilityMask( 2 ),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            RAY_TYPE_RADIANCE,                   // SBT offset
            RAY_TYPE_COUNT,                   // SBT stride
            RAY_TYPE_RADIANCE,                   // missSBTIndex
            p0, p1, p2, p3, p4, reinterpret_cast<uint32_t &>(depth));
    payload_rgb.x = int_as_float(p0);
    payload_rgb.y = int_as_float(p1);
    payload_rgb.z = int_as_float(p2);
    payload_rgb.w = int_as_float(p3);

    return payload_rgb;
}

static __device__ __inline__ float4 traceRefractionRay(
        OptixTraversableHandle handle,
        float3 origin,
        float3 direction,
        float refractionIndex,
        unsigned int depth) {
    unsigned int ray_flags;
    if (optixIsTriangleBackFaceHit()) {
        ray_flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    } else {
        ray_flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES;
    }

    float4 payload_rgb = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
    uint32_t p0, p1, p2, p3, p4;
    p0 = float_as_int(payload_rgb.x);
    p1 = float_as_int(payload_rgb.y);
    p2 = float_as_int(payload_rgb.z);
    p3 = float_as_int(payload_rgb.w);
    p4 = float_as_int(refractionIndex);
    optixTrace(
            handle,
            origin,
            direction,
            1.e-4f,  // tmin
            1e16f,  // tmax
            0.0f,                // rayTime
            OptixVisibilityMask( 2 ),
            ray_flags,
            RAY_TYPE_RADIANCE,                   // SBT offset
            RAY_TYPE_COUNT,                   // SBT stride
            RAY_TYPE_RADIANCE,                   // missSBTIndex
            p0, p1, p2, p3, p4, reinterpret_cast<uint32_t &>(depth));
    payload_rgb.x = int_as_float(p0);
    payload_rgb.y = int_as_float(p1);
    payload_rgb.z = int_as_float(p2);
    payload_rgb.w = int_as_float(p3);

    return payload_rgb;
}