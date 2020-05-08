#pragma once

#include "Device.h"

class CommandBuffer
{
public:
    CommandBuffer(Device& device) : device{device} {};
    VkResult begin();
    void     end();

public:
    Device&         device;
    VkCommandBuffer handle{VK_NULL_HANDLE};
};
