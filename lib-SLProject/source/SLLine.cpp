//
// Created by nic on 26.12.19.
//

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_HAS_OPTIX
#    include <SLMaterial.h>
#    include <SLLine.h>
#    include <SLOptixDefinitions.h>

//-----------------------------------------------------------------------------
SLLine::SLLine(SLAssetManager* assetMgr,
               SLVec3f         p1,
               SLVec3f         p2,
               SLMaterial*     mat) : SLMesh(assetMgr, "line")
{
    _p1.set(p1);
    _p2.set(p2);
    _mat = mat;
}
//-----------------------------------------------------------------------------
void SLLine::createMeshAccelerationStructure()
{
    // Build custom GAS
    uint32_t _buildInput_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

    OptixAabb aabb = {
      min(_p1.x, _p2.x),
      min(_p1.y, _p2.y),
      min(_p1.z, _p2.z),
      max(_p1.x, _p2.x),
      max(_p1.y, _p2.y),
      max(_p1.z, _p2.z),
    };

    _aabb.alloc_and_upload(&aabb, 1);
    CUdeviceptr aabbs[1] = {_aabb.devicePointer()};

    _buildInput.type                                  = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    _buildInput.aabbArray.aabbBuffers                 = aabbs;
    _buildInput.aabbArray.numPrimitives               = 1;
    _buildInput.aabbArray.flags                       = _buildInput_flags;
    _buildInput.aabbArray.numSbtRecords               = 1;
    _buildInput.aabbArray.sbtIndexOffsetBuffer        = 0;
    _buildInput.aabbArray.sbtIndexOffsetSizeInBytes   = 0;
    _buildInput.aabbArray.sbtIndexOffsetStrideInBytes = 0;

    _sbtIndex = RAY_TYPE_COUNT * meshIndex++;

    buildAccelerationStructure();
}
//-----------------------------------------------------------------------------
void SLLine::updateMeshAccelerationStructure()
{
}
//-----------------------------------------------------------------------------
ortHitData SLLine::createHitData()
{
    ortHitData hitData          = SLMesh::createHitData();
    hitData.geometry.line.p1 = make_float3(_p1);
    hitData.geometry.line.p2 = make_float3(_p2);

    return hitData;
}
//-----------------------------------------------------------------------------
void SLLine::init(SLNode* node)
{
}
//-----------------------------------------------------------------------------
#endif // SL_HAS_OPTIX
