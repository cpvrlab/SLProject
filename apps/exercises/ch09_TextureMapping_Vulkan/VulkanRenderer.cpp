#include "VulkanRenderer.h"

VulkanRenderer::~VulkanRenderer()
{
    _device->waitIdle();

    _depthImage->destroy();
    delete _depthImage;
    _framebuffer->destroy();
    delete _framebuffer;
    _commandBuffer->destroy();
    delete _commandBuffer;

    _renderPass->destroy();
    delete _renderPass;
    _swapchain->destroy();
    delete _swapchain;

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

    _descriptorSetLayout->destroy();
    delete _descriptorSetLayout;

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
    _device->destroy();
    delete _device;
    _instance->destroy();
    delete _instance;
}
//-----------------------------------------------------------------------------
VulkanRenderer::VulkanRenderer(GLFWwindow* window)
{
    const vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
    const vector<const char*> deviceExtensions = {"VK_KHR_swapchain", "VK_KHR_maintenance1"};
    // Setting up vulkan
    _instance = new Instance("Test", deviceExtensions, validationLayers);
    glfwCreateWindowSurface(_instance->handle, window, nullptr, &_surface);
    _device     = new Device(*_instance, _instance->physicalDevice, _surface, deviceExtensions);
    _swapchain  = new Swapchain(*_device, window);
    _renderPass = new RenderPass(*_device, *_swapchain);
    _depthImage = new TextureImage(*_device);
    _depthImage->createDepthImage(*_swapchain);
    _framebuffer = new Framebuffer(*_device, *_renderPass, *_swapchain, *_depthImage);
}
//-----------------------------------------------------------------------------
// TODO: Break this method down into diffenent create*()
#include <RangeManager.h>
void VulkanRenderer::createMesh(Camera& camera, const vector<DrawingObject>& drawingObj)
{
    vector<int> indexSize;
    _descriptorSetLayout      = new DescriptorSetLayout(*_device);
    RangeManager rangeManager = RangeManager(drawingObj.size());

    for (int i = 0; i < drawingObj.size(); i++)
    {
        // Shader program setup
        GPUProgram*   program          = drawingObj[i].mat->program();
        ShaderModule* vertShaderModule = new ShaderModule(*_device, program->shaders()[0]->code());
        ShaderModule* fragShaderModule = new ShaderModule(*_device, program->shaders()[1]->code());
        vertShaderModuleList.push_back(vertShaderModule);
        fragShaderModuleList.push_back(fragShaderModule);
        Pipeline* pipeline = new Pipeline(*_device, *_swapchain, *_descriptorSetLayout, *_renderPass, *vertShaderModule, *fragShaderModule);
        pipelineList.push_back(pipeline);

        // Texture setup
        Texture*      tex          = drawingObj[i].mat->textures()[0];
        TextureImage* textureImage = new TextureImage(*_device);
        textureImage->createTextureImage(tex->imageData(), tex->imageWidth(), tex->imageHeight());
        textureImageList.push_back(textureImage);

        int meshCounter = 0;
        // Mesh setup
        for (int j = 0; j < drawingObj[i].nodeList.size(); j++)
        {
            const Mesh* mesh        = drawingObj[i].nodeList[j]->mesh();
            Buffer*     indexBuffer = new Buffer(*_device);
            indexBuffer->createIndexBuffer(mesh->I32);
            indexBufferList.push_back(indexBuffer);
            UniformBuffer* uniformBuffer = new UniformBuffer(*_device, *_swapchain, camera, drawingObj[i].nodeList[j]->om());
            uniformBufferList.push_back(uniformBuffer);

            DescriptorPool* descriptorPool = new DescriptorPool(*_device, *_swapchain);
            descriptorPoolList.push_back(descriptorPool);
            DescriptorSet* descriptorSet = new DescriptorSet(*_device, *_swapchain, *_descriptorSetLayout, *descriptorPool, *uniformBuffer, textureImage->sampler(), *textureImage);
            descriptorSetList.push_back(descriptorSet);
            Buffer* vertexBuffer = new Buffer(*_device);
            vertexBuffer->createVertexBuffer(mesh->P, mesh->N, mesh->Tc, mesh->C, mesh->P.size());
            vertexBufferList.push_back(vertexBuffer);
            indexSize.push_back((int)mesh->I32.size());

            meshCounter++;
        }

        rangeManager.add(i, meshCounter);
    }

    _commandBuffer = new CommandBuffer(*_device);
    _commandBuffer->setVertices(*_swapchain, *_framebuffer, *_renderPass, vertexBufferList, indexBufferList, pipelineList, descriptorSetList, indexSize, rangeManager);
    _device->createSyncObjects(*_swapchain);
}
//-----------------------------------------------------------------------------
void VulkanRenderer::draw()
{
    vkWaitForFences(_device->handle(),
                    1,
                    &_device->inFlightFences()[_currentFrame],
                    VK_TRUE,
                    UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(_device->handle(),
                                            _swapchain->handle(),
                                            UINT64_MAX,
                                            _device->imageAvailableSemaphores()[_currentFrame],
                                            VK_NULL_HANDLE,
                                            &imageIndex);
    if (result != VK_SUCCESS)
        cerr << "failed to acquire swapchain image!" << endl;

    for (UniformBuffer* ub : uniformBufferList)
        ub->update(imageIndex);

    if (_device->imagesInFlight()[imageIndex] != VK_NULL_HANDLE)
        vkWaitForFences(_device->handle(), 1, &_device->imagesInFlight()[imageIndex], VK_TRUE, UINT64_MAX);
    _device->imagesInFlight()[imageIndex] = _device->inFlightFences()[_currentFrame];

    VkSemaphore          waitSemaphores[] = {_device->imageAvailableSemaphores()[_currentFrame]};
    VkPipelineStageFlags waitStages[]     = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSubmitInfo         submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores    = waitSemaphores;
    submitInfo.pWaitDstStageMask  = waitStages;
    submitInfo.commandBufferCount = 1;
    VkCommandBuffer c             = _commandBuffer->handles()[imageIndex];
    submitInfo.pCommandBuffers    = &c;

    VkSemaphore signalSemaphores[]  = {_device->renderFinishedSemaphores()[_currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores    = signalSemaphores;

    vkResetFences(_device->handle(), 1, &_device->inFlightFences()[_currentFrame]);

    result = vkQueueSubmit(_device->graphicsQueue(), 1, &submitInfo, _device->inFlightFences()[_currentFrame]);
    ASSERT_VULKAN(result, "Failed to submit draw command buffer");

    VkSwapchainKHR   swapchains[] = {_swapchain->handle()};
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = signalSemaphores;
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = swapchains;
    presentInfo.pImageIndices      = &imageIndex;

    result = vkQueuePresentKHR(_device->presentQueue(), &presentInfo);
    if (result != VK_SUCCESS)
        cerr << "Failed to present swapchain image" << endl;

    _currentFrame = (_currentFrame + 1) % 2;
}
