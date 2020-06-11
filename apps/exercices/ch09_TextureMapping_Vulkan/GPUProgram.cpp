#include "GPUProgram.h"

GPUProgram::~GPUProgram()
{
    for (GPUShader* shader : _shaders)
        delete shader;
}

void GPUProgram::addShader(GPUShader* shader)
{
    _shaders.push_back(shader);
}
