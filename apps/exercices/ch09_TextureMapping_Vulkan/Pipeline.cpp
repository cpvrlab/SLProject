#include "Pipeline.h"

Pipeline::Pipeline(Device& device, Swapchain& swapchain, DescriptorSetLayout& descriptorSetLayout, RenderPass& renderPass, ShaderModule& vertShaderModule, ShaderModule& fragShaderModule) : _device{device}, _swapchain(swapchain)
{
    createGraphicsPipeline(swapchain.extent(), descriptorSetLayout.handle(), renderPass.handle(), vertShaderModule.handle(), fragShaderModule.handle());
}

void Pipeline::destroy()
{
    if (_graphicsPipeline != VK_NULL_HANDLE) vkDestroyPipeline(_device.handle(), _graphicsPipeline, nullptr);

    if (_pipelineLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(_device.handle(), _pipelineLayout, nullptr);
}

void Pipeline::draw(UniformBuffer& uniformBuffer, CommandBuffer& commandBuffer)
{
    vkWaitForFences(_device.handle(),
                    1,
                    &_device.inFlightFences()[_currentFrame],
                    VK_TRUE,
                    UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(_device.handle(),
                                            _swapchain.handle(),
                                            UINT64_MAX,
                                            _device.imageAvailableSemaphores()[_currentFrame],
                                            VK_NULL_HANDLE,
                                            &imageIndex);
    if (result != VK_SUCCESS)
        cerr << "failed to acquire swapchain image!" << endl;

    uniformBuffer.update(imageIndex);

    if (_device.imagesInFlight()[imageIndex] != VK_NULL_HANDLE)
        vkWaitForFences(_device.handle(), 1, &_device.imagesInFlight()[imageIndex], VK_TRUE, UINT64_MAX);
    _device.imagesInFlight()[imageIndex] = _device.inFlightFences()[_currentFrame];

    VkSemaphore          waitSemaphores[] = {_device.imageAvailableSemaphores()[_currentFrame]};
    VkPipelineStageFlags waitStages[]     = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSubmitInfo         submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores    = waitSemaphores;
    submitInfo.pWaitDstStageMask  = waitStages;
    submitInfo.commandBufferCount = 1;
    VkCommandBuffer c             = commandBuffer.handles()[imageIndex];
    submitInfo.pCommandBuffers    = &c;

    VkSemaphore signalSemaphores[]  = {_device.renderFinishedSemaphores()[_currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores    = signalSemaphores;

    vkResetFences(_device.handle(), 1, &_device.inFlightFences()[_currentFrame]);

    result = vkQueueSubmit(_device.graphicsQueue(), 1, &submitInfo, _device.inFlightFences()[_currentFrame]);
    ASSERT_VULKAN(result, "Failed to submit draw command buffer");

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = signalSemaphores;

    VkSwapchainKHR swapchains[] = {_swapchain.handle()};
    presentInfo.swapchainCount  = 1;
    presentInfo.pSwapchains     = swapchains;

    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(_device.presentQueue(), &presentInfo);
    if (result != VK_SUCCESS)
        cerr << "Failed to present swapchain image" << endl;

    _currentFrame = (_currentFrame + 1) % 2;
}

void Pipeline::createGraphicsPipeline(const VkExtent2D swapchainExtent, const VkDescriptorSetLayout descriptorSetLayout, const VkRenderPass renderPass, const VkShaderModule vertShader, VkShaderModule fragShader)
{
    auto                                 attributeDescriptions = Vertex::getAttributeDescriptions();
    VkVertexInputBindingDescription      bindingDescription    = Vertex::getBindingDescription();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount   = 1;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions      = &bindingDescription;
    vertexInputInfo.pVertexAttributeDescriptions    = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = (float)swapchainExtent.height;
    viewport.width    = (float)swapchainExtent.width;
    viewport.height   = -((float)swapchainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapchainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports    = &viewport;
    viewportState.scissorCount  = 1;
    viewportState.pScissors     = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable        = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth               = 1.0f;
    rasterizer.cullMode                = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable         = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable  = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                          VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT |
                                          VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType             = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable     = VK_FALSE;
    colorBlending.logicOp           = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount   = 1;
    colorBlending.pAttachments      = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts    = &descriptorSetLayout;

    VkResult result = vkCreatePipelineLayout(_device.handle(), &pipelineLayoutInfo, nullptr, &_pipelineLayout);
    ASSERT_VULKAN(result, "Failed to create pipeline layout");

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShader;
    vertShaderStageInfo.pName  = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShader;
    fragShaderStageInfo.pName  = "main";

    VkPipelineShaderStageCreateInfo shaderStages[2];
    shaderStages[0] = vertShaderStageInfo;
    shaderStages[1] = fragShaderStageInfo;
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount          = 2;
    pipelineInfo.pStages             = shaderStages;
    pipelineInfo.pVertexInputState   = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisampling;
    pipelineInfo.pColorBlendState    = &colorBlending;
    pipelineInfo.layout              = _pipelineLayout;
    pipelineInfo.renderPass          = renderPass;
    pipelineInfo.subpass             = 0;
    pipelineInfo.basePipelineHandle  = VK_NULL_HANDLE;

    result = vkCreateGraphicsPipelines(_device.handle(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &_graphicsPipeline);
    ASSERT_VULKAN(result, "Failed to create graphics pipeline");
}
