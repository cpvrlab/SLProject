#pragma once

#include "Buffer.h"
#include "UniformBufferObject.h"
#include <SLMat4.h>
#include <Camera.h>

class Swapchain;
class Device;
class Buffer;

//-----------------------------------------------------------------------------
class UniformBuffer
{
public:
    UniformBuffer(Device& device, Swapchain& swapchain, Camera& camera, SLMat4f& modelPos);
    void destroy();
    void update(uint32_t currentImage);

    // Getter
    vector<Buffer*> buffers() const { return _buffers; }

private:
    Device&         _device;
    Swapchain&      _swapchain;
    vector<Buffer*> _buffers;
    Camera&         _camera;
    SLMat4f&        _modelPos;
};
//-----------------------------------------------------------------------------
