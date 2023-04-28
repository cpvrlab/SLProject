#pragma once

#include "Device.h"
#include <string>

//-----------------------------------------------------------------------------
class ShaderModule
{
public:
    // ShaderModule(Device& device, const string& shaderPath);
    ShaderModule(Device& device, const vector<char>& code);
    void destroy();

    // Getter
    VkShaderModule handle() const { return _handle; }

private:
    void         createShaderModule(const vector<char>& code);
    vector<char> readFile(const string& filename);

    Device&        _device;
    VkShaderModule _handle;
};
//-----------------------------------------------------------------------------
