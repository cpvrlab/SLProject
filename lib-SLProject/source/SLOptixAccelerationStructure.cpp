//
// Created by nic on 01.11.19.
//

#include <SLOptixAccelerationStructure.h>

SLOptixAccelerationStructure::SLOptixAccelerationStructure() {
    _buffer = nullptr;
}

SLOptixAccelerationStructure::~SLOptixAccelerationStructure() {
    delete _buffer;
}