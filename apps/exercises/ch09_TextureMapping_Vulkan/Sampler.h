#ifndef SAMPLER_H
#define SAMPLER_H

#include "Device.h"

//-----------------------------------------------------------------------------
class Sampler
{
public:
    Sampler(Device& device);
    void destroy();

    // Getter
    VkSampler handle() const { return _handle; }

private:
    Device&   _device;
    VkSampler _handle{VK_NULL_HANDLE};
};
//-----------------------------------------------------------------------------
#endif
