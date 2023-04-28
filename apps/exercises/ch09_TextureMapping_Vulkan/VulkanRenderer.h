#ifndef VULKANRENDERER_H
#define VULKANRENDERER_H

#define GLFW_INCLUDE_VULKAN
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <string>
#include <CVImage.h> // Image class for image loading
#include <SLVec3.h>
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
#include "Camera.h"
#include "DrawingObject.h"
#include "RangeManager.h"

class VulkanRenderer
{
public:
    VulkanRenderer(GLFWwindow* window);
    ~VulkanRenderer();

    // VulkanRenderer(const VulkanRenderer&) = default;
    // VulkanRenderer& operator=(const VulkanRenderer&) = default;

    void draw();
    // temp
    void createMesh(Camera& camera, const vector<DrawingObject>& drawingObj);

private:
    void createA();

    VkSurfaceKHR         _surface;
    Instance*            _instance            = nullptr;
    Device*              _device              = nullptr;
    Swapchain*           _swapchain           = nullptr;
    RenderPass*          _renderPass          = nullptr;
    Framebuffer*         _framebuffer         = nullptr;
    TextureImage*        _depthImage          = nullptr;
    DescriptorSetLayout* _descriptorSetLayout = nullptr;
    CommandBuffer*       _commandBuffer       = nullptr;

    vector<DescriptorPool*> descriptorPoolList;
    vector<UniformBuffer*>  uniformBufferList;
    vector<DescriptorSet*>  descriptorSetList;
    vector<TextureImage*>   textureImageList;
    vector<Buffer*>         indexBufferList;
    vector<Buffer*>         vertexBufferList;
    vector<ShaderModule*>   vertShaderModuleList;
    vector<ShaderModule*>   fragShaderModuleList;
    vector<Pipeline*>       pipelineList;

    int _currentFrame = 0;
};

#endif
