#ifndef GPUSHADER_H
#define GPUSHADER_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <Object.h>
#include <vkEnums.h>

using namespace std;

class GPUShader : public Object
{
public:
    GPUShader(string GPUShaderName, string filename, ShaderType type);

    // Getters
    const ShaderType    type() { return _type; }
    const vector<char>& code() { return _code; }

protected:
    vector<char> _code;     //!< string of the shader source code
    string       _filename; //!< path and filename of the shader source code
    ShaderType   _type;     //!< type of the shader

private:
    void readFile();
};
//-----------------------------------------------------------------------------
typedef vector<GPUShader*> VGPUShader;
//-----------------------------------------------------------------------------
#endif
