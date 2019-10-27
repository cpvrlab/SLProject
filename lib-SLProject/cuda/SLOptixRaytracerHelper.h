#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>
#include <optix_types.h>
#include <SLOptixDefinitions.h>
#include <SLOptixVectorMath.h>

static __forceinline__ __device__ void setColor(float3 p) {
    optixSetPayload_0(float_as_int(p.x));
    optixSetPayload_1(float_as_int(p.y));
    optixSetPayload_2(float_as_int(p.z));
}

static __forceinline__ __device__ float getRefractionIndex() {
    return int_as_float(optixGetPayload_3());
}

static __forceinline__ __device__ bool isInside() {
    return optixGetPayload_4();
}

static __forceinline__ __device__ unsigned int getDepth() {
    return optixGetPayload_5();
}

static __forceinline__ __device__ void setOcclusion(float occlusion) {
    optixSetPayload_0(float_as_int(occlusion));
}

static __forceinline__ __device__ float getOcclusion() {
    return int_as_float(optixGetPayload_1());
}

__forceinline__ __device__ uchar3 make_color(const float3 &c) {
    return make_uchar3(
            static_cast<uint8_t>( clamp(c.x, 0.0f, 1.0f) * 255.0f ),
            static_cast<uint8_t>( clamp(c.y, 0.0f, 1.0f) * 255.0f ),
            static_cast<uint8_t>( clamp(c.z, 0.0f, 1.0f) * 255.0f )
    );
}

static __device__ __inline__ float3
traceRadianceRay(
        OptixTraversableHandle handle,
        float3 origin,
        float3 direction,
        float refractionIndex,
        bool isInside,
        unsigned int depth) {
    float3 payload_rgb = make_float3(0.5f, 0.5f, 0.5f);
    uint32_t p0, p1, p2, p3, p4;
    p0 = float_as_int(payload_rgb.x);
    p1 = float_as_int(payload_rgb.y);
    p2 = float_as_int(payload_rgb.z);
    p3 = float_as_int(refractionIndex);
    p4 = isInside;
    optixTrace(
            handle,
            origin,
            direction,
            1.e-4f,  // tmin
            1e16f,  // tmax
            0.0f,                // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            RAY_TYPE_RADIANCE,                   // SBT offset
            RAY_TYPE_COUNT,                   // SBT stride
            RAY_TYPE_RADIANCE,                   // missSBTIndex
            p0, p1, p2, p3, p4, reinterpret_cast<uint32_t &>(depth));
    payload_rgb.x = int_as_float(p0);
    payload_rgb.y = int_as_float(p1);
    payload_rgb.z = int_as_float(p2);

    return payload_rgb;
}