#ifndef PIPELINE_H
#define PIPELINE_H

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

//-----------------------------------------------------------------------------
class Pipeline
{
public:
    Pipeline(Device&              device,
             Swapchain&           swapchain,
             DescriptorSetLayout& descriptorSetLayout,
             RenderPass&          renderPass,
             ShaderModule&        vertShaderModule,
             ShaderModule&        fragShaderModule);
    void destroy();

    // Getter
    VkPipeline       graphicsPipeline() const { return _graphicsPipeline; }
    VkPipelineLayout pipelineLayout() const { return _pipelineLayout; }

private:
    void createGraphicsPipeline(const VkExtent2D            swapchainExtent,
                                const VkDescriptorSetLayout descriptorSetLayout,
                                const VkRenderPass          renderPass,
                                const VkShaderModule        vertShader,
                                const VkShaderModule        fragShader);

    Device&          _device;
    Swapchain&       _swapchain;
    VkPipeline       _graphicsPipeline;
    VkPipelineLayout _pipelineLayout;
};
//-----------------------------------------------------------------------------
#endif
