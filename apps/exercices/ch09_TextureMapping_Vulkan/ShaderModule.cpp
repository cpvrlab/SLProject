#include "ShaderModule.h"

//-----------------------------------------------------------------------------
// ShaderModule::ShaderModule(Device&       device,
//                            const string& shaderPath) : _device{device}
// {
//     vector<char> code = readFile(shaderPath);
//     createShaderModule(code);
// }
//-----------------------------------------------------------------------------
ShaderModule::ShaderModule(Device&             device,
                           const vector<char>& code) : _device{device}
{
    createShaderModule(code);
}
//-----------------------------------------------------------------------------
void ShaderModule::destroy()
{
    if (_handle != VK_NULL_HANDLE)
        vkDestroyShaderModule(_device.handle(), _handle, nullptr);
}
//-----------------------------------------------------------------------------
void ShaderModule::createShaderModule(const vector<char>& code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode    = reinterpret_cast<const uint32_t*>(code.data());

    VkResult result = vkCreateShaderModule(_device.handle(),
                                           &createInfo,
                                           nullptr,
                                           &_handle);
    ASSERT_VULKAN(result, "Failed to create shader module");
}
//-----------------------------------------------------------------------------
vector<char> ShaderModule::readFile(const string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
        throw std::runtime_error("failed to open file!");

    size_t       fileSize = (size_t)file.tellg();
    vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}
//-----------------------------------------------------------------------------
