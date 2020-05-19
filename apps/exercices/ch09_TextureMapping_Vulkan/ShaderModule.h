#pragma once

#include "Device.h"
#include <string>

//-----------------------------------------------------------------------------
class ShaderModule
{
public:
    ShaderModule(Device& device, const string& shaderPath);
    void destroy();

    Device&        device;
    VkShaderModule handle;

private:
    void         createShaderModule(const vector<char>& code);
    vector<char> readFile(const string& filename);
};
//-----------------------------------------------------------------------------
