//#############################################################################
//  File:      SLOptixHelper.h
//  Date:      October 2019
//  Authors:   Nic Dorner, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_HAS_OPTIX
#    include <cuda_runtime.h>
#    include <optix.h>
#    include <optix_types.h>
#    include <SLOptixDefinitions.h>
#    include <SLOptixVectorMath.h>

//-----------------------------------------------------------------------------
static __device__ __inline__ float2 getPixelOffset(uint3 idx)
{
    const uint3 dim = optixGetLaunchDimensions();

    return 2.0f * make_float2(
                    static_cast<float>(idx.x) / static_cast<float>(dim.x),
                    static_cast<float>(idx.y) / static_cast<float>(dim.y)) -
           1.0f;
}
//-----------------------------------------------------------------------------
static __forceinline__ __device__ void setColor(float4 p)
{
    optixSetPayload_0(__float_as_int(p.x));
    optixSetPayload_1(__float_as_int(p.y));
    optixSetPayload_2(__float_as_int(p.z));
    optixSetPayload_3(__float_as_int(p.w));
}
//-----------------------------------------------------------------------------
static __forceinline__ __device__ float getRefractionIndex()
{
    return __int_as_float(optixGetPayload_4());
}
//-----------------------------------------------------------------------------
static __forceinline__ __device__ unsigned int getDepth()
{
    return optixGetPayload_5();
}
//-----------------------------------------------------------------------------
static __forceinline__ __device__ unsigned int getRayType()
{
    return optixGetPayload_6();
}
//-----------------------------------------------------------------------------
static __forceinline__ __device__ void setLighted(float lighted)
{
    optixSetPayload_0(__float_as_int(lighted));
}
//-----------------------------------------------------------------------------
static __forceinline__ __device__ float getLighted()
{
    return __int_as_float(optixGetPayload_0());
}
//-----------------------------------------------------------------------------
static __forceinline__ __device__ float lightAttenuation(ortLight light, float dist)
{
    return 1.0f / (light.kc + light.kl * dist + light.kq * dist * dist);
}
//-----------------------------------------------------------------------------
__forceinline__ __device__ float4 make_color(const float4& c)
{
    return make_float4(
      clamp(c.x, 0.0f, 1.0f),
      clamp(c.y, 0.0f, 1.0f),
      clamp(c.z, 0.0f, 1.0f),
      clamp(c.w, 0.0f, 1.0f));
}
//-----------------------------------------------------------------------------
__forceinline__ __device__ float4 gamma_correction(const float4& c, const float gamma)
{
    return make_float4(
      powf(c.x, gamma),
      powf(c.y, gamma),
      powf(c.z, gamma),
      powf(c.w, gamma));
}
//-----------------------------------------------------------------------------
static __device__ __inline__ float3 getNormalVector()
{
    float3 N;

    auto*        rt_data                = reinterpret_cast<ortHitData*>(optixGetSbtDataPointer());
    unsigned int idx                    = optixGetPrimitiveIndex();
    const float2 barycentricCoordinates = optixGetTriangleBarycentrics();
    const float  u                      = barycentricCoordinates.x;
    const float  v                      = barycentricCoordinates.y;

    if (rt_data->normals && rt_data->indices)
    {
        // Interpolate normal vector with barycentric coordinates
        N = (1.f - u - v) *
              rt_data->normals[rt_data->indices[idx].x] +
            u * rt_data->normals[rt_data->indices[idx].y] +
            v * rt_data->normals[rt_data->indices[idx].z];
        N = normalize(optixTransformNormalFromObjectToWorldSpace(N));
    }
    else
    {
        OptixTraversableHandle gas       = optixGetGASTraversableHandle();
        float3                 vertex[3] = {make_float3(0.0f), make_float3(0.0f), make_float3(0.0f)};
        optixGetTriangleVertexData(gas,
                                   idx,
                                   rt_data->sbtIndex,
                                   0,
                                   vertex);
        N = normalize(cross(vertex[1] - vertex[0], vertex[2] - vertex[0]));
    }

    // if a back face was hit then the normal vector is in the opposite direction
    if (optixIsTriangleBackFaceHit())
    {
        N = N * -1;
    }

    return N;
}
//-----------------------------------------------------------------------------
static __device__ __inline__ float4 getTextureColor()
{
    float4 texture_color = make_float4(1.0);

    auto*        rt_data                = reinterpret_cast<ortHitData*>(optixGetSbtDataPointer());
    unsigned int idx                    = optixGetPrimitiveIndex();
    const float2 barycentricCoordinates = optixGetTriangleBarycentrics();
    const float  u                      = barycentricCoordinates.x;
    const float  v                      = barycentricCoordinates.y;

    if (rt_data->textureObject)
    {
        const float2 tc = (1.f - u - v) * rt_data->texCords[rt_data->indices[idx].x] +
                          u * rt_data->texCords[rt_data->indices[idx].y] +
                          v * rt_data->texCords[rt_data->indices[idx].z];
        texture_color = tex2D<float4>(rt_data->textureObject, tc.x, tc.y);
    }

    return texture_color;
}
//-----------------------------------------------------------------------------
static __forceinline__ __device__ void cosine_sample_hemisphere(const float  u1,
                                                                const float  u2,
                                                                const float3 N,
                                                                float3&      p)
{
    float3 discX = normalize(make_float3(1, 1, (-1 * (N.x + N.y) / N.z)));
    float3 discY = normalize(cross(N, discX));

    // Uniformly sample disk.
    const float r   = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p               = r * cosf(phi) * discX +
        r * sinf(phi) * discY;

    // Project up to hemisphere.
    float l = sqrtf(fmaxf(0.0f, 1.0f - length(p) * length(p)));
    p += l * N;
}
//-----------------------------------------------------------------------------
static __device__ __inline__ float4 tracePrimaryRay(
  OptixTraversableHandle handle,
  float3                 origin,
  float3                 direction)
{
    float4   payload_rgb = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
    uint32_t p0, p1, p2, p3, p4, p5, p6;
    // Color payload
    p0 = __float_as_int(payload_rgb.x);
    p1 = __float_as_int(payload_rgb.y);
    p2 = __float_as_int(payload_rgb.z);
    p3 = __float_as_int(payload_rgb.w);
    // Current refraction index
    p4 = __float_as_int(1.0f);
    // Current depth
    p5 = 1;
    // Ray type
    p6 = 0;
    optixTrace(
      handle,
      origin,
      direction,
      1.e-4f, // tmin
      1e16f,  // tmax
      0.0f,   // rayTime
      OptixVisibilityMask(255),
      OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
      RAY_TYPE_RADIANCE, // SBT offset
      RAY_TYPE_COUNT,    // SBT stride
      RAY_TYPE_RADIANCE, // missSBTIndex
      p0,
      p1,
      p2,
      p3,
      p4,
      p5,
      p6);
    // Write the resulting color back to local floating point values
    payload_rgb.x = __int_as_float(p0);
    payload_rgb.y = __int_as_float(p1);
    payload_rgb.z = __int_as_float(p2);
    payload_rgb.w = __int_as_float(p3);

    return payload_rgb;
}
//-----------------------------------------------------------------------------
static __device__ __inline__ float4 traceSecondaryRay(
  OptixTraversableHandle handle,
  float3                 origin,
  float3                 direction)
{
    float4   payload_rgb = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
    uint32_t p0, p1, p2, p3, p4, p5;
    // Color payload
    p0 = __float_as_int(payload_rgb.x);
    p1 = __float_as_int(payload_rgb.y);
    p2 = __float_as_int(payload_rgb.z);
    p3 = __float_as_int(payload_rgb.w);
    // Current refraction index
    p4 = __float_as_int(1.0f);
    // Current depth
    p5 = getDepth() + 1;
    optixTrace(
      handle,
      origin,
      direction,
      1.e-4f, // tmin
      1e16f,  // tmax
      0.0f,   // rayTime
      OptixVisibilityMask(255),
      OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
      RAY_TYPE_RADIANCE, // SBT offset
      RAY_TYPE_COUNT,    // SBT stride
      RAY_TYPE_RADIANCE, // missSBTIndex
      p0,
      p1,
      p2,
      p3,
      p4,
      p5);
    // Write the resulting color back to local floating point values
    payload_rgb.x = __int_as_float(p0);
    payload_rgb.y = __int_as_float(p1);
    payload_rgb.z = __int_as_float(p2);
    payload_rgb.w = __int_as_float(p3);

    return payload_rgb;
}
//-----------------------------------------------------------------------------
static __device__ __inline__ float traceShadowRay(OptixTraversableHandle handle,
                                                  float3                 origin,
                                                  float3                 direction,
                                                  float                  dist)
{
    uint32_t p0 = __float_as_int(1.0f);

    // Send shadow ray
    optixTrace(
      handle,
      origin,
      direction,
      1.e-4f, // tmin
      dist,   // tmax
      0.0f,   // rayTime
      OptixVisibilityMask(1),
      OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
      RAY_TYPE_OCCLUSION, // SBT offset
      RAY_TYPE_COUNT,     // SBT stride
      RAY_TYPE_OCCLUSION, // missSBTIndex
      p0                  // payload
    );

    float lighted = min(__int_as_float(p0), 1.0f);

    return lighted;
}
//-----------------------------------------------------------------------------
static __device__ __inline__ float4 traceReflectionRay(
  OptixTraversableHandle handle,
  float3                 origin,
  float3                 N,
  float3                 ray_dir)
{
    float3 R = reflect(ray_dir, N);

    float4   payload_rgb = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
    uint32_t p0, p1, p2, p3, p4, p5, p6;
    p0 = __float_as_int(payload_rgb.x);
    p1 = __float_as_int(payload_rgb.y);
    p2 = __float_as_int(payload_rgb.z);
    p3 = __float_as_int(payload_rgb.w);
    p4 = __float_as_int(getRefractionIndex());
    p5 = getDepth() + 1;
    p6 = 0;
    optixTrace(
      handle,
      origin,
      R,
      1.e-4f, // tmin
      1e16f,  // tmax
      0.0f,   // rayTime
      OptixVisibilityMask(2),
      OPTIX_RAY_FLAG_DISABLE_ANYHIT,
      RAY_TYPE_RADIANCE, // SBT offset
      RAY_TYPE_COUNT,    // SBT stride
      RAY_TYPE_RADIANCE, // missSBTIndex
      p0,
      p1,
      p2,
      p3,
      p4,
      p5,
      p6);
    payload_rgb.x = __int_as_float(p0);
    payload_rgb.y = __int_as_float(p1);
    payload_rgb.z = __int_as_float(p2);
    payload_rgb.w = __int_as_float(p3);

    return payload_rgb;
}
//-----------------------------------------------------------------------------
static __device__ __inline__ float4 traceRefractionRay(
  OptixTraversableHandle handle,
  float3                 origin,
  float3                 N,
  float3                 ray_dir,
  float                  kn)
{
    // calculate eta
    float refractionIndex = kn;
    float eta             = getRefractionIndex() / kn;
    if (optixIsTriangleBackFaceHit())
    {
        refractionIndex = 1.0f;
        eta             = kn / 1.0f;
    }

    // calculate transmission vector T
    float3 T;
    float  c1 = dot(N, -ray_dir);
    float  w  = eta * c1;
    float  c2 = 1.0f + (w - eta) * (w + eta);
    if (c2 >= 0.0f)
    {
        T = eta * ray_dir + (w - sqrtf(c2)) * N;
    }
    else
    {
        T = 2.0f * (dot(-ray_dir, N)) * N + ray_dir;
    }

    unsigned int ray_flags;
    if (optixIsTriangleBackFaceHit())
    {
        ray_flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    }
    else
    {
        ray_flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES;
    }

    float4   payload_rgb = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
    uint32_t p0, p1, p2, p3, p4, p5, p6;
    p0 = __float_as_int(payload_rgb.x);
    p1 = __float_as_int(payload_rgb.y);
    p2 = __float_as_int(payload_rgb.z);
    p3 = __float_as_int(payload_rgb.w);
    p4 = __float_as_int(refractionIndex);
    p5 = getDepth() + 1;
    p6 = 1;
    optixTrace(
      handle,
      origin,
      T,
      1.e-4f, // tmin
      1e16f,  // tmax
      0.0f,   // rayTime
      OptixVisibilityMask(2),
      ray_flags,
      RAY_TYPE_RADIANCE, // SBT offset
      RAY_TYPE_COUNT,    // SBT stride
      RAY_TYPE_RADIANCE, // missSBTIndex
      p0,
      p1,
      p2,
      p3,
      p4,
      p5,
      p6);
    payload_rgb.x = __int_as_float(p0);
    payload_rgb.y = __int_as_float(p1);
    payload_rgb.z = __int_as_float(p2);
    payload_rgb.w = __int_as_float(p3);

    return payload_rgb;
}
//-----------------------------------------------------------------------------
#endif