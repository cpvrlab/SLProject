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
    unsigned int sbtIndex() const { return _sbtIndex;}

    void optixTraversableHandle(OptixTraversableHandle handle) { _handle = handle; }
    void buffer(SLCudaBuffer<void>*  buffer) { _buffer = buffer; }
    void sbtIndex(unsigned int sbtIndex) { _sbtIndex = sbtIndex; }
private:
    OptixTraversableHandle  _handle; //!< Handle for generated geometry acceleration structure
    SLCudaBuffer<void>*     _buffer;
    unsigned int            _sbtIndex;
};

#endif //SLPROJECT_SLOPTIXACCELERATIONSTRUCTURE_H
