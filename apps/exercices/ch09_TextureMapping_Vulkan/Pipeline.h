#pragma once

#include "Device.h"
#include "Swapchain.h"
#include "DescriptorSetLayout.h"
#include "RenderPass.h"
#include "ShaderModule.h"

class Pipeline
{
public:
    Pipeline(Device& device, Swapchain& swapchain, DescriptorSetLayout& descriptorSetLayout, RenderPass renderPass, ShaderModule& vertShaderModule, ShaderModule& fragShaderModule);

private:
    void createGraphicsPipeline(VkExtent2D swapchainExtent, VkDescriptorSetLayout descriptorSetLayout, VkRenderPass renderPass, VkShaderModule vertShader, VkShaderModule fragShader);

public:
    Device&          device;
    VkPipelineLayout pipelineLayout;
    VkPipeline       graphicsPipeline;
};
