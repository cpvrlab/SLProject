#ifndef IMAGE_H
#define IMAGE_H

#include "Buffer.h"

// forward decalaration
class Buffer;
class CommandBuffer;
class Device;
class Sampler;

//-----------------------------------------------------------------------------
// TODO: Rename to Image + remove sampler + make class Texture for textures (has a image and sampler)
class TextureImage
{
public:
    TextureImage(Device& device) : _device{device} { ; }
    void destroy();

    // Getter
    VkImage     image() const { return _image; }
    VkImageView imageView() const { return _imageView; }
    Sampler&    sampler() const { return *_sampler; }

    // Setter
    // void setSampler(Sampler& sampler) { _sampler = &sampler; }

    void createTextureImage(void*        pixels,
                            unsigned int texWidth,
                            unsigned int texHeight);
    void createDepthImage(Swapchain& swapchain);

private:
    void        createImage(uint32_t              width,
                            uint32_t              height,
                            VkFormat              format,
                            VkImageTiling         tiling,
                            VkImageUsageFlags     usage,
                            VkMemoryPropertyFlags properties);
    void        transitionImageLayout(VkImage&      image,
                                      VkFormat      format,
                                      VkImageLayout oldLayout,
                                      VkImageLayout newLayout);
    void        copyBufferToImage(VkBuffer buffer,
                                  VkImage  image,
                                  uint32_t width,
                                  uint32_t height);
    VkImageView createImageView(VkImage&           image,
                                VkFormat           format,
                                VkImageAspectFlags aspectFlags);

public:
    Device&        _device;
    VkImage        _image;
    VkDeviceMemory _imageMemory;
    VkImageView    _imageView;
    Sampler*       _sampler = nullptr;
};
//-----------------------------------------------------------------------------
#endif
