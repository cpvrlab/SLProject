#include "ShaderModule.h"

ShaderModule::ShaderModule(Device& device, const string& shaderPath) : device{device}
{
    std::vector<char> code = readFile(shaderPath);
    createShaderModule(code);
}

void ShaderModule::destroy()
{
    if (handle != VK_NULL_HANDLE)
        vkDestroyShaderModule(device.handle, handle, nullptr);
}

void ShaderModule::createShaderModule(const std::vector<char>& code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode    = reinterpret_cast<const uint32_t*>(code.data());

    VkResult result = vkCreateShaderModule(device.handle, &createInfo, nullptr, &handle);
    ASSERT_VULKAN(result, "Failed to create shader module");
}

vector<char> ShaderModule::readFile(const string& filename)
{
    ifstream file(filename, ios::ate | ios::binary);

    if (!file.is_open())
        throw runtime_error("failed to open file!");

    size_t       fileSize = (size_t)file.tellg();
    vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}
