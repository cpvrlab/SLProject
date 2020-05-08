#pragma once

#include "Device.h"

// forward decalaration
class Buffer;

class TextureImage
{
public:
    TextureImage(Device& device, void* pixels, unsigned int texWidth, unsigned int texHeight);

private:
    void        createImage(uint32_t              width,
                            uint32_t              height,
                            VkFormat              format,
                            VkImageTiling         tiling,
                            VkImageUsageFlags     usage,
                            VkMemoryPropertyFlags properties,
                            VkImage&              image);
    void        transitionImageLayout(VkImage       image,
                                      VkFormat      format,
                                      VkImageLayout oldLayout,
                                      VkImageLayout newLayout);
    void        copyBufferToImage(VkBuffer buffer,
                                  VkImage  image,
                                  uint32_t width,
                                  uint32_t height);
    VkImageView createImageView(VkImage image, VkFormat format);

public:
    Device&     device;
    VkImage     image;
    VkImageView textureImageView;
    Buffer*     buffer;
};
