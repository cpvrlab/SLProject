#pragma once

#include "Buffer.h"

// forward decalaration
class Buffer;
class CommandBuffer;
class Device;

//-----------------------------------------------------------------------------
class TextureImage
{
public:
    TextureImage(Device&      device,
                 void*        pixels,
                 unsigned int texWidth,
                 unsigned int texHeight);
    void destroy();

private:
    void        createImage(uint32_t              width,
                            uint32_t              height,
                            VkFormat              format,
                            VkImageTiling         tiling,
                            VkImageUsageFlags     usage,
                            VkMemoryPropertyFlags properties,
                            VkImage&              image,
                            Buffer*               buffer);
    void        transitionImageLayout(VkImage&      image,
                                      VkFormat      format,
                                      VkImageLayout oldLayout,
                                      VkImageLayout newLayout);
    void        copyBufferToImage(VkBuffer buffer,
                                  VkImage  image,
                                  uint32_t width,
                                  uint32_t height);
    VkImageView createImageView(VkImage& image, VkFormat format);

public:
    Device&        device;
    VkImage        image;
    VkDeviceMemory imageMemory;
    VkImageView    textureImageView;
    // Buffer*     buffer;
};
//-----------------------------------------------------------------------------
