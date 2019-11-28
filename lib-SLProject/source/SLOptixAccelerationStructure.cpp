//
// Created by nic on 01.11.19.
//

#include <SLOptixAccelerationStructure.h>
#include <SL/SLApplication.h>

SLOptixAccelerationStructure::SLOptixAccelerationStructure() {
    _accelBuildOptions.buildFlags   =   OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                        OPTIX_BUILD_FLAG_ALLOW_UPDATE |
                                        OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS |
                                        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    _buffer = new SLCudaBuffer<void>();
}

SLOptixAccelerationStructure::~SLOptixAccelerationStructure() {
    delete _buffer;
}

void SLOptixAccelerationStructure::buildAccelerationStructure() {
    OptixDeviceContext context = SLApplication::context;

    _accelBuildOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OPTIX_CHECK( optixAccelComputeMemoryUsage(
            context,
            &_accelBuildOptions,
            &_buildInput,
            1,  // num_build_inputs
            &_accelBufferSizes
    ) );

    SLCudaBuffer<void> temp_buffer = SLCudaBuffer<void>();
    temp_buffer.alloc(_accelBufferSizes.tempSizeInBytes);

    // non-compacted output
    auto* non_compacted_output_buffer = new SLCudaBuffer<void>();
    non_compacted_output_buffer->alloc(_accelBufferSizes.outputSizeInBytes);

    SLCudaBuffer<OptixAabb> aabbBuffer = SLCudaBuffer<OptixAabb>();
    aabbBuffer.alloc(sizeof(OptixAabb));
    SLCudaBuffer<size_t> compactedSize = SLCudaBuffer<size_t>();
    compactedSize.alloc(sizeof(size_t));

    OptixAccelEmitDesc emitProperty[2];
    emitProperty[0].type               = OPTIX_PROPERTY_TYPE_AABBS;
    emitProperty[0].result             = aabbBuffer.devicePointer();
    emitProperty[1].type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty[1].result             = compactedSize.devicePointer();

    OPTIX_CHECK( optixAccelBuild(
            context,
            SLApplication::stream,             // CUDA stream
            &_accelBuildOptions,
            &_buildInput,
            1,                                 // num build inputs
            temp_buffer.devicePointer(),
            _accelBufferSizes.tempSizeInBytes,
            non_compacted_output_buffer->devicePointer(),
            _accelBufferSizes.outputSizeInBytes,
            &_handle,
            emitProperty,                      // emitted property list
            2                                  // num emitted properties
    ) );
    CUDA_SYNC_CHECK( SLApplication::stream );

    OptixAabb aabb;
    aabbBuffer.download(&aabb);

    size_t compacted_accel_size;
    compactedSize.download(&compacted_accel_size);

    if(compacted_accel_size < _accelBufferSizes.outputSizeInBytes )
    {
        auto* outputBuffer = new SLCudaBuffer<void>();
        outputBuffer->alloc(compacted_accel_size );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact(context,
                                       SLApplication::stream,
                                       _handle,
                                       outputBuffer->devicePointer(),
                                       compacted_accel_size,
                                       &_handle ) );
        CUDA_SYNC_CHECK( SLApplication::stream );

        delete non_compacted_output_buffer;
        _buffer = outputBuffer;
    } else {
        _buffer = non_compacted_output_buffer;
    }
}

void SLOptixAccelerationStructure::updateAccelerationStructure() {
    OptixDeviceContext context = SLApplication::context;

    _accelBuildOptions.operation              = OPTIX_BUILD_OPERATION_UPDATE;

    SLCudaBuffer<void> temp_buffer = SLCudaBuffer<void>();
    temp_buffer.alloc(_accelBufferSizes.tempUpdateSizeInBytes);

    OPTIX_CHECK( optixAccelBuild(
            context,
            SLApplication::stream,             // CUDA stream
            &_accelBuildOptions,
            &_buildInput,
            1,                                 // num build inputs
            temp_buffer.devicePointer(),
            _accelBufferSizes.tempUpdateSizeInBytes,
            _buffer->devicePointer(),
            _accelBufferSizes.outputSizeInBytes,
            &_handle,
            nullptr,                      // emitted property list
            0                                  // num emitted properties
    ) );
    CUDA_SYNC_CHECK( SLApplication::stream );
}
