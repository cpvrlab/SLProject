#pragma once

#include "DescriptorSetLayout.h"
#include "RenderPass.h"
#include "ShaderModule.h"
#include "UniformBuffer.h"
#include "Vertex.cpp"

struct Vertex;
class Device;
class Swapchain;
class CommandBuffer;
class UniformBuffer;

class Pipeline
{
public:
    Pipeline(Device& device, Swapchain& swapchain, DescriptorSetLayout& descriptorSetLayout, RenderPass& renderPass, ShaderModule& vertShaderModule, ShaderModule& fragShaderModule);
    void destroy();

    void draw(Swapchain& swapchain, UniformBuffer& uniformBuffer, CommandBuffer& commandBuffer);

private:
    void createGraphicsPipeline(VkExtent2D swapchainExtent, VkDescriptorSetLayout descriptorSetLayout, VkRenderPass renderPass, VkShaderModule vertShader, VkShaderModule fragShader);

public:
    Device&          device;
    VkPipelineLayout pipelineLayout;
    VkPipeline       graphicsPipeline;
    int              currentFrame = 0;
};
