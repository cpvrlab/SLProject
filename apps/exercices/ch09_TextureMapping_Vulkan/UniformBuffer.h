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
    UniformBuffer(Device& device, Swapchain& swapchain, SLMat4f& camera, SLMat4f& modelPos);
    void destroy();
    void update(uint32_t currentImage);

    // Getter
    vector<Buffer*> buffers() const { return _buffers; }

private:
    Device&         _device;
    Swapchain&      _swapchain;
    vector<Buffer*> _buffers;
    SLMat4f&        _camera;
    SLMat4f&        _modelPos;
};
//-----------------------------------------------------------------------------
