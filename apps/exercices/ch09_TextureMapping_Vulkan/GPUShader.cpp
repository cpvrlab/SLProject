#include "GPUShader.h"

GPUShader::GPUShader(string GPUShaderName, string filename, ShaderType type)
  : Object(GPUShaderName), _filename(filename), _type(type)
{
    readFile();
}

void GPUShader::readFile()
{
    ifstream file(_filename.c_str(), ios::ate | ios::binary);

    if (!file.is_open())
        throw runtime_error("Failed to open file!");

    file.seekg(0, ios::end);
    _code.reserve(file.tellg());
    file.seekg(0, ios::beg);

    _code.assign((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());

    file.close();
}
