//
// Created by nic on 01.11.19.
//

#ifndef SLPROJECT_SLOPTIXACCELERATIONSTRUCTURE_H
#define SLPROJECT_SLOPTIXACCELERATIONSTRUCTURE_H

#include <optix_types.h>
#include <SLCudaBuffer.h>

class SLOptixAccelerationStructure
{
public:
    SLOptixAccelerationStructure();
    ~SLOptixAccelerationStructure();

    OptixTraversableHandle optixTraversableHandle() { return _handle; }
protected:
    void buildAccelerationStructure(OptixBuildInput);
    OptixTraversableHandle  _handle = 0;         //!< Handle for generated geometry acceleration structure
    SLCudaBuffer<void>*     _buffer = nullptr;
};

#endif //SLPROJECT_SLOPTIXACCELERATIONSTRUCTURE_H
