#include "VulkanRenderer.h"

VulkanRenderer::~VulkanRenderer()
{
    device->waitIdle();

    framebuffer->destroy();
    commandBuffer->destroy();
    pipeline->destroy();
    renderPass->destroy();
    swapchain->destroy();
    uniformBuffer->destroy();
    descriptorPool->destroy();
    descriptorSetLayout->destroy();
    textureImage->destroy();
    indexBuffer->destroy();
    vertexBuffer->destroy();
    vertShaderModule->destroy();
    fragShaderModule->destroy();
    device->destroy();
    instance->destroy();

    delete device;
    delete instance;
    delete framebuffer;
    delete commandBuffer;
    delete pipeline;
    delete renderPass;
    delete swapchain;
    delete uniformBuffer;
    delete descriptorPool;
    delete descriptorSetLayout;
    delete textureImage;
    delete indexBuffer;
    delete vertexBuffer;
    delete fragShaderModule;
    delete vertShaderModule;
}

VulkanRenderer::VulkanRenderer(GLFWwindow* window)
{
    const vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
    const vector<const char*> deviceExtensions = {"VK_KHR_swapchain", "VK_KHR_maintenance1"};
    // Setting up vulkan
    instance = new Instance("Test", deviceExtensions, validationLayers);
    glfwCreateWindowSurface(instance->handle, window, nullptr, &surface);
    device     = new Device(*instance, instance->physicalDevice, surface, deviceExtensions);
    swapchain  = new Swapchain(*device, window);
    renderPass = new RenderPass(*device, *swapchain);
}

void VulkanRenderer::createMesh(SLMat4f& camera, SLMat4f& modelPos, Mesh& mesh)
{
    // Shader program setup
    descriptorSetLayout = new DescriptorSetLayout(*device);
    vertShaderModule    = new ShaderModule(*device, vertShaderPath);
    fragShaderModule    = new ShaderModule(*device, fragShaderPath);
    pipeline            = new Pipeline(*device, *swapchain, *descriptorSetLayout, *renderPass, *vertShaderModule, *fragShaderModule);
    framebuffer         = new Framebuffer(*device, *renderPass, *swapchain);

    // Texture setup
    textureImage = new TextureImage(*device, mesh.mat->textures()[0]->imageData(), mesh.mat->textures()[0]->imageWidth(), mesh.mat->textures()[0]->imageHeight());

    // Mesh setup
    indexBuffer = new Buffer(*device);
    indexBuffer->createIndexBuffer(mesh.I32);
    uniformBuffer  = new UniformBuffer(*device, *swapchain, camera, modelPos);
    descriptorPool = new DescriptorPool(*device, *swapchain);
    descriptorSet  = new DescriptorSet(*device, *swapchain, *descriptorSetLayout, *descriptorPool, *uniformBuffer, textureImage->sampler(), *textureImage);
    vertexBuffer   = new Buffer(*device);
    vertexBuffer->createVertexBuffer(mesh.P, mesh.N, mesh.Tc, mesh.C, mesh.P.size());
    // Draw call setup
    commandBuffer = new CommandBuffer(*device);
    commandBuffer->setVertices(*swapchain, *framebuffer, *renderPass, *vertexBuffer, *indexBuffer, *pipeline, *descriptorSet, (int)mesh.I32.size());
    device->createSyncObjects(*swapchain);
}

void VulkanRenderer::draw()
{
    pipeline->draw(*uniformBuffer, *commandBuffer);
}
