#pragma once

#include "Device.h"
#include "Buffer.h"

class IndexBuffer
{
public:
    IndexBuffer(Device& device, const std::vector<uint16_t> indices);

public:
    Device& device;
    Buffer* buffer;
};
