#ifndef VULKANRENDERER_H
#define VULKANRENDERER_H

#define GLFW_INCLUDE_VULKAN
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <string>
#include <CVImage.h> // Image class for image loading
#include <math/SLVec3.h>
#include <glUtils.h>
#include <Utils.h>

#include "Instance.h"
#include "Device.h"
#include "Swapchain.h"
#include "RenderPass.h"
#include "DescriptorSetLayout.h"
#include "ShaderModule.h"
#include "Pipeline.h"
#include "Framebuffer.h"
#include "TextureImage.h"
#include "Sampler.h"
#include "IndexBuffer.h"
#include "UniformBuffer.h"
#include "DescriptorPool.h"
#include "DescriptorSet.h"
#include "VertexBuffer.h"
#include "Node.h"
#include <memory>
#include "DrawingObject.h"

class VulkanRenderer
{
public:
    VulkanRenderer(GLFWwindow* window);
    ~VulkanRenderer();

    // VulkanRenderer(const VulkanRenderer&) = default;
    // VulkanRenderer& operator=(const VulkanRenderer&) = default;

    void draw();
    // temp
    void createMesh(SLMat4f& camera, const vector<DrawingObject>& drawingObj);

private:
    void createA();

    VkSurfaceKHR  surface;
    Instance*     instance    = nullptr;
    Device*       device      = nullptr;
    Swapchain*    swapchain   = nullptr;
    RenderPass*   renderPass  = nullptr;
    Framebuffer*  framebuffer = nullptr;
    TextureImage* depthImage  = nullptr;

    vector<CommandBuffer*>       commandBufferList;
    vector<UniformBuffer*>       uniformBufferList;
    vector<DescriptorPool*>      descriptorPoolList;
    vector<DescriptorSet*>       descriptorSetList;
    vector<DescriptorSetLayout*> descriptorSetLayoutList;
    vector<TextureImage*>        textureImageList;
    vector<Buffer*>              indexBufferList;
    vector<Buffer*>              vertexBufferList;
    vector<ShaderModule*>        vertShaderModuleList;
    vector<ShaderModule*>        fragShaderModuleList;
    vector<Pipeline*>            pipelineList;
};

#endif
