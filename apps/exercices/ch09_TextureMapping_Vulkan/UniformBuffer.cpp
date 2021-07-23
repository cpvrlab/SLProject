#include "UniformBuffer.h"

//-----------------------------------------------------------------------------
UniformBuffer::UniformBuffer(Device&    device,
                             Swapchain& swapchain,
                             Camera&    camera,
                             SLMat4f&   modelPos) : _device{device},
                                                  _swapchain{swapchain},
                                                  _camera{camera},
                                                  _modelPos{modelPos}
{
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    _buffers.resize(_swapchain.images().size());
    _uniformAllocInfo.resize(_swapchain.images().size());

    for (size_t i = 0; i < _swapchain.images().size(); i++)
    {
        _buffers[i] = new Buffer(_device);
        _buffers[i]->createBuffer(bufferSize,
                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  &_uniformAllocInfo[i]);
    }
}
//-----------------------------------------------------------------------------
void UniformBuffer::destroy()
{
    for (size_t i = 0; i < _buffers.size(); i++)
        if (_buffers[i] != nullptr)
        {
            _buffers[i]->destroy();
            delete (_buffers[i]);
        }
}
//-----------------------------------------------------------------------------
void UniformBuffer::update(uint32_t currentImage)
{
    ubo.model    = _modelPos;
    ubo.view     = _camera.om();
    float width  = (float)_swapchain.extent().width;
    float height = (float)_swapchain.extent().height;
    ubo.proj.perspective(_camera.fov(),
                         _camera.viewportRatio(),
                         _camera.clipNear(),
                         _camera.clipFar());

    for (size_t i = 0; i < _uniformAllocInfo.size(); i++)
    {
        memcpy(_uniformAllocInfo[i].pMappedData, &ubo, sizeof(ubo));
    }
}
//-----------------------------------------------------------------------------
