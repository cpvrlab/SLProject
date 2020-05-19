#pragma once

#include "Buffer.h"
#include "UniformBufferObject.h"
#include <math/SLMat4.h>

class Swapchain;
class Device;
class Buffer;

//-----------------------------------------------------------------------------
class UniformBuffer
{
public:
    UniformBuffer(Device& device, Swapchain& swapchain, SLMat4f& camera);
    void destroy();
    void update(uint32_t currentImage);

    Device&         device;
    Swapchain&      swapchain;
    vector<Buffer*> buffers;
    SLMat4f&        camera;
};
//-----------------------------------------------------------------------------