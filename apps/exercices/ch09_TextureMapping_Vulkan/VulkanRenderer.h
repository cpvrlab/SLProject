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

class VulkanRenderer
{
public:
    VulkanRenderer(GLFWwindow* window);
    ~VulkanRenderer();

    // VulkanRenderer(const VulkanRenderer&) = default;
    // VulkanRenderer& operator=(const VulkanRenderer&) = default;

    void draw();
    // temp
    void createMesh(SLMat4f& camera, SLMat4f& modelPos, Mesh& mesh);

private:
    string vertShaderPath = SLstring(SL_PROJECT_ROOT) + "/data/shaders/vertShader.vert.spv";
    string fragShaderPath = SLstring(SL_PROJECT_ROOT) + "/data/shaders/fragShader.frag.spv";

    Device*              device = nullptr;
    VkSurfaceKHR         surface;
    Instance*            instance            = nullptr;
    Framebuffer*         framebuffer         = nullptr;
    CommandBuffer*       commandBuffer       = nullptr;
    Pipeline*            pipeline            = nullptr;
    RenderPass*          renderPass          = nullptr;
    Swapchain*           swapchain           = nullptr;
    UniformBuffer*       uniformBuffer       = nullptr;
    DescriptorPool*      descriptorPool      = nullptr;
    DescriptorSet*       descriptorSet       = nullptr;
    DescriptorSetLayout* descriptorSetLayout = nullptr;
    TextureImage*        textureImage        = nullptr;
    Buffer*              indexBuffer         = nullptr;
    Buffer*              vertexBuffer        = nullptr;
    ShaderModule*        vertShaderModule    = nullptr;
    ShaderModule*        fragShaderModule    = nullptr;
};

#endif
