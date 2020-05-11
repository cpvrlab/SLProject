#pragma once

#include "Device.h"
#include "Swapchain.h"
#include "DescriptorSetLayout.h"
#include "DescriptorPool.h"
#include "UniformBuffer.h"
#include "Sampler.h"
#include "TextureImage.h"

#include <vector>

class DescriptorSet
{
public:
    DescriptorSet(Device&, Swapchain&, DescriptorSetLayout&, DescriptorPool&, UniformBuffer&, Sampler&, TextureImage&);

public:
    Device&                      device;
    std::vector<VkDescriptorSet> handles;
};
