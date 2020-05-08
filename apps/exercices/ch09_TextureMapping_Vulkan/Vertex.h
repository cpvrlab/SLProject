#include "math/SLVec3.h"
#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"
#include <array>

struct Vertex
{
    SLVec2f pos;
    SLVec3f color;
    SLVec3f texCoord;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding                         = 0;
        bindingDescription.stride                          = sizeof(Vertex);
        bindingDescription.inputRate                       = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions()
    {
        array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};

        attributeDescriptions[0].binding  = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format   = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset   = offsetof(Vertex, pos);

        attributeDescriptions[1].binding  = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset   = offsetof(Vertex, color);

        attributeDescriptions[2].binding  = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format   = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset   = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }
};
