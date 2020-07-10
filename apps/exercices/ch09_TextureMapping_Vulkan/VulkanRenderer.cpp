#include "VulkanRenderer.h"

VulkanRenderer::~VulkanRenderer()
{
    device->waitIdle();

    depthImage->destroy();
    delete depthImage;
    framebuffer->destroy();
    delete framebuffer;
    for (CommandBuffer* c : commandBufferList)
    {
        c->destroy();
        delete c;
    }
    renderPass->destroy();
    delete renderPass;
    swapchain->destroy();
    delete swapchain;
    for (Pipeline* p : pipelineList)
    {
        p->destroy();
        delete p;
    }
    for (UniformBuffer* u : uniformBufferList)
    {
        u->destroy();
        delete u;
    }
    for (DescriptorPool* d : descriptorPoolList)
    {
        d->destroy();
        delete d;
    }
    {
        descriptorSetLayout->destroy();
        delete descriptorSetLayout;
    }
    for (TextureImage* t : textureImageList)
    {
        t->destroy();
        delete t;
    }
    for (Buffer* i : indexBufferList)
    {
        i->destroy();
        delete i;
    }
    for (Buffer* v : vertexBufferList)
    {
        v->destroy();
        delete v;
    }
    for (ShaderModule* v : vertShaderModuleList)
    {
        v->destroy();
        delete v;
    }
    for (ShaderModule* f : fragShaderModuleList)
    {
        f->destroy();
        delete f;
    }
    device->destroy();
    delete device;
    instance->destroy();
    delete instance;
}
//-----------------------------------------------------------------------------
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
    depthImage = new TextureImage(*device);
    depthImage->createDepthImage(*swapchain);
    framebuffer = new Framebuffer(*device, *renderPass, *swapchain, *depthImage);
}
//-----------------------------------------------------------------------------
void VulkanRenderer::createMesh(SLMat4f& camera, const vector<DrawingObject>& drawingObj)
{
    descriptorSetLayout = new DescriptorSetLayout(*device);

    for (int i = 0; i < drawingObj.size(); i++)
    {
        // Shader program setup
        GPUProgram*   program          = drawingObj[1].mat->program();
        ShaderModule* vertShaderModule = new ShaderModule(*device, program->shaders()[0]->code());
        ShaderModule* fragShaderModule = new ShaderModule(*device, program->shaders()[1]->code());
        vertShaderModuleList.push_back(vertShaderModule);
        fragShaderModuleList.push_back(fragShaderModule);
        Pipeline* pipeline = new Pipeline(*device, *swapchain, *descriptorSetLayout, *renderPass, *vertShaderModule, *fragShaderModule);
        pipelineList.push_back(pipeline);

        // Texture setup
        Texture*      tex          = drawingObj[0].mat->textures()[0];
        TextureImage* textureImage = new TextureImage(*device);
        textureImage->createTextureImage(tex->imageData(), tex->imageWidth(), tex->imageHeight());
        textureImageList.push_back(textureImage);

        // Mesh setup
        const Mesh* mesh        = drawingObj[0].nodeList[0]->mesh();
        Buffer*     indexBuffer = new Buffer(*device);
        indexBuffer->createIndexBuffer(mesh->I32);
        indexBufferList.push_back(indexBuffer);
        UniformBuffer* uniformBuffer = new UniformBuffer(*device, *swapchain, camera, drawingObj[1].nodeList[0]->om());
        uniformBufferList.push_back(uniformBuffer);
        DescriptorPool* descriptorPool = new DescriptorPool(*device, *swapchain);
        descriptorPoolList.push_back(descriptorPool);

        DescriptorSet* descriptorSet = new DescriptorSet(*device, *swapchain, *descriptorSetLayout, *descriptorPool, *uniformBuffer, textureImage->sampler(), *textureImage);
        descriptorSetList.push_back(descriptorSet);
        Buffer* vertexBuffer = new Buffer(*device);
        vertexBuffer->createVertexBuffer(mesh->P, mesh->N, mesh->Tc, mesh->C, mesh->P.size());
        vertexBufferList.push_back(vertexBuffer);
        // Draw call setup
        CommandBuffer* commandBuffer = new CommandBuffer(*device);
        commandBuffer->setVertices(*swapchain, *framebuffer, *renderPass, *vertexBuffer, *indexBuffer, *pipeline, *descriptorSet, (int)mesh->I32.size());
        commandBufferList.push_back(commandBuffer);
        device->createSyncObjects(*swapchain);
    }
}
//-----------------------------------------------------------------------------
void VulkanRenderer::draw()
{
    for (int i = 0; i < pipelineList.size(); i++)
        pipelineList[i]->draw(*uniformBufferList[i], *commandBufferList[i]);
}
