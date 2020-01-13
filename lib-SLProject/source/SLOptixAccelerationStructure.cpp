//#############################################################################
//  File:      SLOptixAccelerationStructure.cpp
//  Author:    Nic Dorner
//  Date:      October 2019
//  Copyright: Nic Dorner
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_HAS_OPTIX

#    include <SLOptixAccelerationStructure.h>
#    include <SL/SLApplication.h>

//-----------------------------------------------------------------------------
SLOptixAccelerationStructure::SLOptixAccelerationStructure()
{
    _accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE |
                                    OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS |
                                    OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    _buffer = new SLCudaBuffer<void>();
}
//-----------------------------------------------------------------------------
SLOptixAccelerationStructure::~SLOptixAccelerationStructure()
{
    delete _buffer;
}
//-----------------------------------------------------------------------------
void SLOptixAccelerationStructure::buildAccelerationStructure()
{
    OptixDeviceContext context = SLApplication::context;

    _accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OPTIX_CHECK(optixAccelComputeMemoryUsage(
      context,
      &_accelBuildOptions,
      &_buildInput,
      1, // num_build_inputs
      &_accelBufferSizes));

    SLCudaBuffer<void> temp_buffer = SLCudaBuffer<void>();
    temp_buffer.alloc(_accelBufferSizes.tempSizeInBytes);

    // non-compacted output
    _buffer = new SLCudaBuffer<void>();
    _buffer->alloc(_accelBufferSizes.outputSizeInBytes);

    SLCudaBuffer<OptixAabb> aabbBuffer = SLCudaBuffer<OptixAabb>();
    aabbBuffer.alloc(sizeof(OptixAabb));
    SLCudaBuffer<size_t> compactedSize = SLCudaBuffer<size_t>();
    compactedSize.alloc(sizeof(size_t));

    OptixAccelEmitDesc emitProperty[1];
    emitProperty[0].type   = OPTIX_PROPERTY_TYPE_AABBS;
    emitProperty[0].result = aabbBuffer.devicePointer();

    OPTIX_CHECK(optixAccelBuild(
      context,
      SLApplication::stream, // CUDA stream
      &_accelBuildOptions,
      &_buildInput,
      1, // num build inputs
      temp_buffer.devicePointer(),
      _accelBufferSizes.tempSizeInBytes,
      _buffer->devicePointer(),
      _accelBufferSizes.outputSizeInBytes,
      &_handle,
      emitProperty, // emitted property list
      1             // num emitted properties
      ));
    CUDA_SYNC_CHECK(SLApplication::stream);

    OptixAabb aabb;
    aabbBuffer.download(&aabb);
}
//-----------------------------------------------------------------------------
void SLOptixAccelerationStructure::updateAccelerationStructure()
{
    OptixDeviceContext context = SLApplication::context;

    _accelBuildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;

    SLCudaBuffer<void> temp_buffer = SLCudaBuffer<void>();
    temp_buffer.alloc(_accelBufferSizes.tempUpdateSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
      context,
      SLApplication::stream, // CUDA stream
      &_accelBuildOptions,
      &_buildInput,
      1, // num build inputs
      temp_buffer.devicePointer(),
      _accelBufferSizes.tempUpdateSizeInBytes,
      _buffer->devicePointer(),
      _accelBufferSizes.outputSizeInBytes,
      &_handle,
      nullptr, // emitted property list
      0        // num emitted properties
      ));
}
//-----------------------------------------------------------------------------
#endif // SL_HAS_OPTIX