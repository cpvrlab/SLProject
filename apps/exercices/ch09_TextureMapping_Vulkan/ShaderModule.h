#pragma once

#include "Device.h"
#include <string>

class ShaderModule
{
public:
    ShaderModule(Device& device, const string& shaderPath);
    void destroy();

private:
    void         createShaderModule(const std::vector<char>& code);
    vector<char> readFile(const string& filename);

public:
    Device&        device;
    VkShaderModule handle;
};
