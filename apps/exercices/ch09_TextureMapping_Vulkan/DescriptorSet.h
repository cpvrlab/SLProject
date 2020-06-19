#pragma once

#include "DescriptorSetLayout.h"
#include "DescriptorPool.h"
#include "UniformBuffer.h"
#include "Sampler.h"
#include "TextureImage.h"

class Device;
class Swapchain;
class UniformBuffer;
class TextureImage;

//-----------------------------------------------------------------------------
class DescriptorSet
{
public:
    DescriptorSet(Device&,
                  Swapchain&,
                  DescriptorSetLayout&,
                  DescriptorPool&,
                  UniformBuffer&,
                  Sampler&,
                  TextureImage&);

    // Getter
    vector<VkDescriptorSet> handles() const { return _handles; }

private:
    Device&                 _device;
    vector<VkDescriptorSet> _handles;
};
//-----------------------------------------------------------------------------
