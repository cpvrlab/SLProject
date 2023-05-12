#include "TextureImage.h"

//-----------------------------------------------------------------------------
void TextureImage::destroy()
{
    if (_imageView != VK_NULL_HANDLE)
        vkDestroyImageView(_device.handle(), _imageView, nullptr);

    if (_image != VK_NULL_HANDLE)
    {
        vkDestroyImage(_device.handle(), _image, nullptr);
        vkFreeMemory(_device.handle(), _imageMemory, nullptr);
    }

    if (_sampler)
    {
        _sampler->destroy();
        free(_sampler);
    }
}
void TextureImage::createTextureImage(void* pixels, unsigned int texWidth, unsigned int texHeight)
{
    VkDeviceSize imageSize = texWidth * texHeight * 4; // * 4 because of RGBA

    if (texWidth == 0)
        cerr << "Failed to load texture _image" << endl;

    Buffer buffer = Buffer(_device);
    buffer.createBuffer(imageSize,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* data;
    vkMapMemory(_device.handle(), buffer.memory(), 0, imageSize, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(_device.handle(), buffer.memory());

    createImage(texWidth,
                texHeight,
                VK_FORMAT_R8G8B8A8_SRGB,
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    transitionImageLayout(_image,
                          VK_FORMAT_R8G8B8A8_SRGB,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(buffer.handle(),
                      _image,
                      static_cast<uint32_t>(texWidth),
                      static_cast<uint32_t>(texHeight));
    transitionImageLayout(_image,
                          VK_FORMAT_R8G8B8A8_SRGB,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    buffer.destroy();

    _imageView = createImageView(_image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
    _sampler   = new Sampler(_device);
}
void TextureImage::createDepthImage(Swapchain& swapchain)
{
    const VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
    createImage(swapchain.extent().width,
                swapchain.extent().height,
                depthFormat,
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // TODO: Find a better solution (without return)
    _imageView = createImageView(_image, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
}
//-----------------------------------------------------------------------------
void TextureImage::createImage(uint32_t              width,
                               uint32_t              height,
                               VkFormat              format,
                               VkImageTiling         tiling,
                               VkImageUsageFlags     usage,
                               VkMemoryPropertyFlags properties)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width  = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth  = 1;
    imageInfo.mipLevels     = 1;
    imageInfo.arrayLayers   = 1;
    imageInfo.format        = format;
    imageInfo.tiling        = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage         = usage;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateImage(_device.handle(), &imageInfo, nullptr, &_image);
    ASSERT_VULKAN(result, "Failed to create _image");

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(_device.handle(), _image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize  = memRequirements.size;
    allocInfo.memoryTypeIndex = Buffer::findMemoryType(_device, memRequirements.memoryTypeBits, properties);

    result = vkAllocateMemory(_device.handle(), &allocInfo, nullptr, &_imageMemory);
    ASSERT_VULKAN(result, "Failed to allocate _image memory");

    vkBindImageMemory(_device.handle(), _image, _imageMemory, 0);
}
//-----------------------------------------------------------------------------
void TextureImage::transitionImageLayout(VkImage&      _image,
                                         VkFormat      format,
                                         VkImageLayout oldLayout,
                                         VkImageLayout newLayout)
{
    CommandBuffer commandBuffer = CommandBuffer(_device);
    commandBuffer.begin();

    VkImageMemoryBarrier barrier{};
    barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout                       = oldLayout;
    barrier.newLayout                       = newLayout;
    barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.image                           = _image;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = 1;

    VkPipelineStageFlags sourceStage{};
    VkPipelineStageFlags destinationStage{};

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage      = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else
        cerr << "Unsupported layout transition!" << endl;

    vkCmdPipelineBarrier(commandBuffer.handle(),
                         sourceStage,
                         destinationStage,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         1,
                         &barrier);
    commandBuffer.end();
}
//-----------------------------------------------------------------------------
void TextureImage::copyBufferToImage(VkBuffer buffer,
                                     VkImage  _image,
                                     uint32_t width,
                                     uint32_t height)
{
    CommandBuffer commandBuffer = CommandBuffer(_device);
    commandBuffer.begin();

    VkBufferImageCopy imageCopyBuffer{};
    imageCopyBuffer.bufferOffset                    = 0;
    imageCopyBuffer.bufferRowLength                 = 0;
    imageCopyBuffer.bufferImageHeight               = 0;
    imageCopyBuffer.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    imageCopyBuffer.imageSubresource.mipLevel       = 0;
    imageCopyBuffer.imageSubresource.baseArrayLayer = 0;
    imageCopyBuffer.imageSubresource.layerCount     = 1;
    imageCopyBuffer.imageOffset                     = {0, 0, 0};
    imageCopyBuffer.imageExtent                     = {width, height, 1};

    vkCmdCopyBufferToImage(commandBuffer.handle(),
                           buffer,
                           _image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1,
                           &imageCopyBuffer);
    commandBuffer.end();
}
//-----------------------------------------------------------------------------
VkImageView TextureImage::createImageView(VkImage&           _image,
                                          VkFormat           format,
                                          VkImageAspectFlags aspectFlags)
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = _image;
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format                          = format;
    viewInfo.subresourceRange.aspectMask     = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    VkImageView imageView;

    VkResult result = vkCreateImageView(_device.handle(), &viewInfo, nullptr, &imageView);
    ASSERT_VULKAN(result, "Failed to create texture _image view!");

    return imageView;
}
//-----------------------------------------------------------------------------
