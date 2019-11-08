#include <SLOptixRaytracerHelper.h>

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __miss__radiance()
{
    auto* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    setColor(rt_data->bg_color);
}

extern "C" __global__ void __miss__occlusion()
{
}

extern "C" __global__ void __anyhit__radiance()
{
}

extern "C" __global__ void __anyhit__occlusion()
{
    auto *rt_data = reinterpret_cast<HitData *>( optixGetSbtDataPointer());
    float occlusion = getOcclusion();
    setOcclusion(occlusion + (1.0f - rt_data->material.kt));
}

extern "C" __global__ void __closesthit__radiance() {
    // Get all data for the hit point
    auto *rt_data = reinterpret_cast<HitData *>( optixGetSbtDataPointer());
    Material material = rt_data->material;
    unsigned int idx = optixGetPrimitiveIndex();
    const float3 ray_dir = optixGetWorldRayDirection();

    // calculate normal vector
    float3 N_0;
    if (rt_data->normals && rt_data->indices) {
        const float2 barycentricCoordinates = optixGetTriangleBarycentrics();
        const float u = barycentricCoordinates.x;
        const float v = barycentricCoordinates.y;
        N_0 = (1.f-u-v) * rt_data->normals[rt_data->indices[idx].x]
              +         u * rt_data->normals[rt_data->indices[idx].y]
              +         v * rt_data->normals[rt_data->indices[idx].z];
    } else {
        OptixTraversableHandle gas = optixGetGASTraversableHandle();
        float3 vertex[3] = { make_float3(0.0f), make_float3(0.0f), make_float3(0.0f)};
        optixGetTriangleVertexData(gas,
                                   idx,
                                   rt_data->sbtIndex,
                                   0,
                                   vertex);
        float transform[12];
        optixGetObjectToWorldTransformMatrix(transform);
        N_0 = normalize(cross(vertex[1] - vertex[0], vertex[2] - vertex[0]));
    }


    // calculate normal vector and hit point
    const float3 N = faceforward(N_0, -ray_dir, N_0);
//    const float3 N = normalize( optixTransformNormalFromObjectToWorldSpace( N_0) );
    const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    float4 color = make_float4(0.0f);
    // calculate local illumination for every light source
    for (int i = 0; i < params.numLights; i++) {
        const float Ldist = length(params.lights[i].position - P);
        const float3 L = normalize(params.lights[i].position - P);
        const float nDl = dot(N, L);

        const float3 R = normalize(reflect(-L, N));
        const float3 V = normalize(-ray_dir);

        float occlusion = 0.0f;
        uint32_t p0 = float_as_int( occlusion );
        if ( nDl > 0.0f)
        {
            // Send shadow ray
            optixTrace(
                    params.handle,
                    P,
                    L,
                    1e-3f,  // tmin
                    Ldist,  // tmax
                    0.0f,                // rayTime
                    OptixVisibilityMask( 1 ),
                    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                    RAY_TYPE_OCCLUSION,                   // SBT offset
                    RAY_TYPE_COUNT,                   // SBT stride
                    RAY_TYPE_OCCLUSION,                   // missSBTIndex
                    p0 // payload
            );
        }

        occlusion = int_as_float( p0 );
        occlusion = min(occlusion, 1.0f);

        // Shading like whitted
        color = color +
                (material.specular_color * powf( max(dot(R, V), 0.0), material.shininess ) * (1.0f - occlusion) // specular
                + material.diffuse_color * max(nDl, 0.0f) * (1.0f - occlusion)) // diffuse
                * params.lights[i].color; // multiply with light color
    }

    // Send reflection ray
    float4 reflection = make_float4(0.0f);
    if(getDepth() < params.max_depth && material.kr > 0.0f) {
        reflection = traceRadianceRay(params.handle, P, reflect(ray_dir, N), getRefractionIndex(), isInside(), getDepth() + 1);
    }

    // Send refraction ray
    float4 transmission = make_float4(0.0f);
    if(getDepth() < params.max_depth && material.kt > 0.0f) {
        // calculate eta
        float refractionIndex = material.kn;
        float eta = getRefractionIndex() / material.kn;
        if (optixIsTriangleBackFaceHit()) {
            refractionIndex = getRefractionIndex();
            eta = material.kn / getRefractionIndex();
        }

        // calculate transmission vector T
        float3 T;
        float c1 = dot(N, -ray_dir);
        float w = eta * c1;
        float c2 = 1.0f + (w - eta) * (w + eta);
        if(c2 >= 0.0f) {
            T = eta * ray_dir + (w - sqrtf(c2)) * N;
        } else {
            T = 2.0f * (dot(-ray_dir, N)) * N  + ray_dir;
        }
        transmission = traceRadianceRay(params.handle, P, T, refractionIndex, isInside(), getDepth() + 1);
    }

    // Make sure the color value does not exceed 1
    color.x = min(color.x, 1.0f);
    color.y = min(color.y, 1.0f);
    color.z = min(color.z, 1.0f);

    // Add together local material shading + ambient color of the material + reflection + transmission
    setColor((color + // local
              material.ambient_color + // ambient color
              reflection * material.kr) * (1.0f - material.kt) + // reflection
             transmission * material.kt); // transmission
}